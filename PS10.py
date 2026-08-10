"""
PS10.py -- Bayesian Belief Network for Water Potability
BITS Pilani, AIMLCZG557/AECLZG557 -- Assignment 2, PS10

Trains a discrete Bayesian Belief Network (BBN) on water_potability.csv and
answers arbitrary posterior queries of the form
    P(Potability | some subset of the chemical attributes)
read from an input file, writing the results to an output file.

Design summary (see design document for full discussion):
  * Continuous attributes are discretized into 3 states (Low/Medium/High)
    using quantile bin edges fitted on the training data.
  * Network structure ("expert" mode, the default): Potability is modeled
    as the common cause of every chemical reading, i.e. edges
    Potability -> ph, Potability -> Hardness, ... (a naive-Bayes-shaped
    BBN). This keeps every attribute's CPD small (one parent) so parameter
    estimates stay stable with a modest amount of data, and it lets the
    inference engine marginalize over ANY subset of missing attributes,
    which the assignment's varying evidence sets require.
  * An alternate structure ("learned" mode) is also implemented: a
    Hill-Climb search over BIC score discovers dependencies directly from
    the data instead of assuming the naive-Bayes shape. Select it with
    --structure learned. See the design document for the accuracy /
    interpretability / runtime trade-offs between the two.
  * CPDs are fit with a Bayesian estimator (BDeu prior) rather than raw
    maximum likelihood, so state combinations that are rare or absent in
    the training data don't collapse to a zero/undefined probability.

Input file format (inputPSXX.txt):
  One or more query blocks. Each block is exactly two non-blank lines:
    1. a comma-separated header naming attributes, e.g.
         ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Turbidity
    2. a comma-separated row of values aligned with that header, e.g.
         3.72,204.89,20791.32,7.3,368.5,564.30,2.96
  Values may be raw numbers (which are discretized using the fitted bin
  edges) or qualitative labels ("Low"/"Medium"/"High", case-insensitive) --
  this covers both the numeric-evidence queries and the descriptive
  low/high-style query in the assignment brief. Any subset/ordering of the
  9 attribute names is accepted. A `Potability` column, if present, is
  treated as a reference label for display only -- it is never used as
  evidence (conditioning on the target would make the query circular).
  Blank lines separate blocks; lines starting with '#' are comments.

Usage:
    uv run PS10.py [--data water_potability.csv] [--input inputPS10.txt]
                    [--output outputPS10.txt] [--structure expert|learned]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# pgmpy 1.1.x emits FutureWarnings about internal reorganizations (e.g.
# StructureScore/HillClimbSearch moving to pgmpy.causal_discovery) that are
# not yet stable replacements; suppressed here so they don't clutter
# grading output. Functionality is unaffected on the pinned pgmpy version
# (see pyproject.toml / uv.lock).
warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")

from pgmpy.estimators import BIC, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET_COLUMN = "Potability"
STATES = ["Low", "Medium", "High"]
N_BINS = len(STATES)


class DataPreparationError(Exception):
    """Raised when the training dataset can't be loaded or prepared."""


class InputFormatError(Exception):
    """Raised when the query input file doesn't follow the expected format."""


class EvidenceError(Exception):
    """Raised when a single query's evidence can't be resolved against the model."""


# --------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise DataPreparationError(f"Training data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise DataPreparationError(f"Dataset is missing expected column(s): {missing_cols}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Median-imputes missing values in the feature columns (fit on this same data)."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        df[col] = df[col].fillna(df[col].median())
    return df


def fit_bin_edges(df: pd.DataFrame) -> dict[str, list[float]]:
    """Fits quantile bin edges per feature so evidence values (raw numbers) can
    later be mapped onto the same Low/Medium/High states used to train the CPDs.
    The outer edges are widened to +/-inf so out-of-range evidence at query time
    still falls into the nearest state instead of being dropped as missing."""
    edges: dict[str, list[float]] = {}
    for col in FEATURE_COLUMNS:
        try:
            _, bins = pd.qcut(df[col], q=N_BINS, retbins=True, duplicates="drop")
        except ValueError as exc:
            raise DataPreparationError(f"Could not discretize column '{col}': {exc}") from exc
        interior = list(bins[1:-1])
        edges[col] = [-np.inf, *interior, np.inf]
    return edges


def discretize_series(series: pd.Series, col: str, edges: dict[str, list[float]]) -> pd.Series:
    n_states = len(edges[col]) - 1
    labels = STATES[:n_states] if n_states == len(STATES) else [f"Bin{i}" for i in range(n_states)]
    return pd.cut(series, bins=edges[col], labels=labels, include_lowest=True)


def discretize_dataframe(df: pd.DataFrame, edges: dict[str, list[float]]) -> pd.DataFrame:
    disc = df.copy()
    for col in FEATURE_COLUMNS:
        disc[col] = discretize_series(df[col], col, edges).astype(str)
    disc[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return disc


# --------------------------------------------------------------------------
# Network construction
# --------------------------------------------------------------------------

def build_expert_structure() -> list[tuple[str, str]]:
    """Naive-Bayes-shaped BBN: Potability -> every chemical attribute."""
    return [(TARGET_COLUMN, col) for col in FEATURE_COLUMNS]


def build_learned_structure(disc_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Alternate structure: BIC-scored Hill-Climb search over the discretized data."""
    hc = HillClimbSearch(disc_df)
    dag = hc.estimate(scoring_method=BIC(disc_df), show_progress=False)
    edges = list(dag.edges())
    if not any(TARGET_COLUMN in edge for edge in edges):
        # Guarantee Potability stays connected to the network even if the
        # search found it conditionally independent of everything else.
        edges.append((TARGET_COLUMN, FEATURE_COLUMNS[0]))
    return edges


def fit_network(disc_df: pd.DataFrame, edges: list[tuple[str, str]]) -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork(edges)
    model.fit(
        disc_df,
        estimator=DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5),
    )
    return model


# --------------------------------------------------------------------------
# Evidence normalization & inference
# --------------------------------------------------------------------------

def _to_state(col: str, raw_value: str, edges: dict[str, list[float]]) -> str:
    text = str(raw_value).strip()
    for state in STATES:
        if text.lower() == state.lower():
            return state
    try:
        numeric = float(text)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(
            f"Value '{raw_value}' for '{col}' is neither numeric nor one of {STATES}"
        ) from exc
    state = discretize_series(pd.Series([numeric]), col, edges).iloc[0]
    if pd.isna(state):
        raise EvidenceError(f"Value '{raw_value}' for '{col}' could not be discretized")
    return str(state)


def normalize_evidence(
    raw_evidence: dict[str, str], edges: dict[str, list[float]]
) -> tuple[dict[str, str], list[tuple[str, str, str]], str | None]:
    """Maps a raw {column: value} query row onto discrete network states.

    Returns (evidence_for_query, display_rows, reference_potability_label).
    `Potability`, if present in the row, is stripped out and returned as a
    reference label only -- never used to condition the query.
    """
    lookup = {name.lower(): name for name in [*FEATURE_COLUMNS, TARGET_COLUMN]}
    evidence: dict[str, str] = {}
    display: list[tuple[str, str, str]] = []
    reference = None
    for raw_col, raw_value in raw_evidence.items():
        canonical = lookup.get(raw_col.strip().lower())
        if canonical is None:
            raise EvidenceError(f"Unknown attribute '{raw_col}' in evidence")
        if canonical == TARGET_COLUMN:
            reference = raw_value
            continue
        state = _to_state(canonical, raw_value, edges)
        evidence[canonical] = state
        display.append((canonical, raw_value, state))
    if not evidence:
        raise EvidenceError("Query supplied no usable evidence attributes")
    return evidence, display, reference


def infer_potability(
    model: DiscreteBayesianNetwork, evidence: dict[str, str]
) -> tuple[dict[int, float], list[str]]:
    """Runs exact inference (Variable Elimination) for P(Potability | evidence).

    Evidence attributes absent from the fitted network (e.g. a feature the
    structure-learning search left disconnected) are ignored rather than
    raising an error, and reported back to the caller so it can be surfaced
    to the user.
    """
    ve = VariableElimination(model)
    valid_evidence = {k: v for k, v in evidence.items() if k in model.nodes()}
    ignored = sorted(set(evidence) - set(valid_evidence))
    result = ve.query(variables=[TARGET_COLUMN], evidence=valid_evidence, show_progress=False)
    order = result.state_names[TARGET_COLUMN]
    probs = {int(state): float(p) for state, p in zip(order, result.values)}
    return probs, ignored


# --------------------------------------------------------------------------
# Input / output file handling
# --------------------------------------------------------------------------

def parse_input_file(path: str) -> list[dict[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        raise InputFormatError(f"Input file not found: {file_path}")
    raw_lines = file_path.read_text().splitlines()
    lines = [ln for ln in raw_lines if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) % 2 != 0:
        raise InputFormatError(
            "Input file must contain header/value line pairs; found an odd number of data lines"
        )
    queries: list[dict[str, str]] = []
    for i in range(0, len(lines), 2):
        header = [h.strip() for h in lines[i].split(",")]
        values = [v.strip() for v in lines[i + 1].split(",")]
        if len(header) != len(values):
            raise InputFormatError(
                f"Header/value column count mismatch around line {i + 1}: "
                f"{len(header)} header(s) vs {len(values)} value(s)"
            )
        queries.append(dict(zip(header, values)))
    if not queries:
        raise InputFormatError("Input file contained no queries")
    return queries


def format_result(
    display_rows: list[tuple[str, str, str]],
    probs: dict[int, float],
    reference: str | None,
    ignored: list[str],
) -> str:
    lines = []
    lines.append("Evidence:")
    for col, raw_value, state in display_rows:
        lines.append(f"  {col} = {raw_value}  ->  {state}")
    if reference is not None:
        lines.append(f"  (Potability={reference} was given in the input as a reference label; not used as evidence)")
    if ignored:
        lines.append(f"  (Ignored -- not part of the learned network: {', '.join(ignored)})")
    lines.append("")

    p0 = probs.get(0, 0.0)
    p1 = probs.get(1, 0.0)
    lines.append(f"{'Potability':<14}{'0':<15}{'1':<15}")
    lines.append(f"{'Probability':<14}{p0:<15.6f}{p1:<15.6f}")
    lines.append("")
    lines.append(f"Probability of the water being good (Potability=1) is {p1 * 100:.2f}%.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------

def build_model(data_path: str, structure: str) -> tuple[DiscreteBayesianNetwork, dict[str, list[float]]]:
    raw_df = load_dataset(data_path)
    imputed_df = impute_missing(raw_df)
    edges_bins = fit_bin_edges(imputed_df)
    disc_df = discretize_dataframe(imputed_df, edges_bins)

    structure_edges = (
        build_expert_structure() if structure == "expert" else build_learned_structure(disc_df)
    )
    model = fit_network(disc_df, structure_edges)
    return model, edges_bins


def run(data_path: str, input_path: str, output_path: str, structure: str) -> int:
    try:
        model, bin_edges = build_model(data_path, structure)
        queries = parse_input_file(input_path)
    except (DataPreparationError, InputFormatError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    sections = []
    for i, raw_evidence in enumerate(queries, start=1):
        try:
            evidence, display_rows, reference = normalize_evidence(raw_evidence, bin_edges)
            probs, ignored = infer_potability(model, evidence)
        except EvidenceError as exc:
            sections.append(f"Query {i}: ERROR -- {exc}")
            continue
        sections.append(f"Query {i}\n" + format_result(display_rows, probs, reference, ignored))

    separator = "\n\n" + ("-" * 60) + "\n\n"
    Path(output_path).write_text(separator.join(sections) + "\n")
    print(f"Wrote {len(queries)} result(s) to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bayesian Belief Network for water potability (PS10).")
    parser.add_argument("--data", default="water_potability.csv", help="Training CSV path")
    parser.add_argument("--input", default="inputPS10.txt", help="Query input file")
    parser.add_argument("--output", default="outputPS10.txt", help="Where to write results")
    parser.add_argument(
        "--structure",
        choices=["expert", "learned"],
        default="expert",
        help="expert = naive-Bayes-shaped DAG (default); learned = Hill-Climb/BIC structure search",
    )
    args = parser.parse_args(argv)
    return run(args.data, args.input, args.output, args.structure)


if __name__ == "__main__":
    raise SystemExit(main())
