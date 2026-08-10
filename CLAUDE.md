# Assignment 2 — PS10: Water Quality (Bayesian Belief Network)

BITS Pilani, AIMLCZG557/AECLZG557, S2 2025-2026. Weightage 13%. **Deadline: 17 Aug 2026, 11:55 PM IST.**
Group assignment — one submission only, no resubmission. Full spec in [Assignment 2 PS10.pdf](Assignment%202%20PS10.pdf).

## Problem

Build a Bayesian Belief Network (BBN) over `water_potability.csv` (3276 rows, 9 numeric features +
binary `Potability` target) and use it to answer inference queries. Attributes:

`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity` → `Potability` (0/1).

The dataset has missing values in several columns (visible already in row 2: `ph` blank; `Sulfate` blank
in rows 3–4) — must be handled explicitly (imputation or a "missing" state), and documented as a design
decision, not silently dropped without comment.

## Required capabilities (map directly to functions/CLI queries)

1. **Construct the BBN** from the CSV — structure (DAG) + CPDs (parameters).
2. **Predict potability** given a partial evidence row (e.g. `ph, Hardness, Solids, Chloramines, Sulfate,
   Conductivity, Turbidity` — note `Organic_carbon` and `Trihalomethanes` absent). Output: full posterior
   `P(Potability=0), P(Potability=1)`.
3. **Infer probability** given a different partial evidence row (e.g. all features except `ph`). The
   sample input tables include a `Potability` column with a value (the row's *true* label from the
   dataset) — **this is NOT evidence to condition on**. Feeding it in would make the query trivial
   (P=1 for the given value). Treat it as reference/ground-truth shown for comparison only; the query
   still infers `P(Potability | the other given features)`.
4. **Qualitative query**: P(Potability=1) given categorical/descriptive evidence — "low pH, high
   hardness, high solids, other chemicals high." Requires discretizing continuous variables into
   states (e.g. Low/Medium/High via tertiles or domain thresholds) so the query terms map onto BBN
   node states.

Because queries 2–4 each supply **different, variable-length subsets of evidence columns**, the
inference function must accept an arbitrary evidence dict and marginalize over everything else
automatically (this is what a BBN buys you over a fixed-input classifier) — don't hardcode which
columns are present.

## Environment & package management

Use **`uv`** for everything Python in this project — no bare `pip`/`venv`/`conda`.

- Init once: `uv init --no-readme` (creates `pyproject.toml`) if one doesn't exist yet.
- Add deps: `uv add pgmpy pandas numpy` (add `openpyxl` too if any tooling touches the `.xlsx`
  contribution file).
- Run the script: `uv run python <script>.py` (or `uv run <script>.py` with a shebang) — always run
  through `uv run` so it resolves against the project's locked environment, never a system/global
  interpreter.
- Commit `pyproject.toml` and `uv.lock` so the environment is reproducible; note in the design doc that
  `uv sync` recreates it exactly.
- Reminder: the deliverable is still a **single `.py` file** with the implementation — `uv`/`pyproject.toml`
  only manage the environment, they don't change the "don't fragment code into multiple files" rule.
  Do not zip `.venv`, `pyproject.toml`, or `uv.lock` into the submission unless the spec's deliverables
  list is amended to ask for them — as written it only lists the `.py` file, so plan to have the
  grader's environment cover installing `pgmpy`/`pandas`/`numpy` some other way (e.g. note the
  `uv add` commands in the design doc) unless told otherwise.

## Suggested technical approach

- **Library**: `pgmpy` is the standard fit here — `DiscreteBayesianNetwork`, `MaximumLikelihoodEstimator`
  / `BayesianEstimator` for CPDs, `VariableElimination` for exact inference. Install via
  `uv add pgmpy` (see Environment section above) rather than `pip install`.
- **Discretization**: BBN nodes need discrete states. Bin each continuous feature (e.g. into 3 bins:
  Low/Medium/High via quantiles, or WHO-guideline-informed thresholds where sensible — e.g. pH). Keep
  bin edges as fitted parameters (from training data), not hardcoded literals, so the model generalizes
  to a different evaluation CSV.
- **Structure**: either (a) hand-designed DAG based on domain reasoning (e.g. chemical attributes →
  Potability, roughly naive-Bayes-with-target-as-child, or a richer expert graph), or (b) a structure-
  learning algorithm (Hill-Climb search with BIC/K2 score in pgmpy). Pick one, justify it in the design
  doc, and use the **other** as the "alternate modeling approach + performance implications" the PDF
  requires in the design document.
- **Missing data**: impute (mean/median per column, or a learned CPD-compatible "missing" bin) *before*
  discretization — document the choice and its effect on distributions.
- **Inference**: exact inference via Variable Elimination is fine at this scale; note in the design doc
  that exact inference doesn't scale to large/dense networks and approximate methods (e.g. sampling)
  would be the alternative — ties into the "alternate approach" requirement.

## Hard constraints from the spec (do not violate)

- **No hardcoded values** — script must read the training CSV path and `inputPS10.txt` generically;
  grading uses a different CSV/input file with the same schema/syntax.
- **Single `.py` file** — do not fragment code into modules.
- Code must be **modular and well-documented**, with basic **error handling** (e.g. missing file,
  malformed row, empty/out-of-range values) surfaced as clear messages.
- Any debug/print-the-data-structure helper functions must be **commented out before submission**.
- Output written to `outputPS10.txt` in the exact table format shown in the sample output (Potability
  0/1 header row, Probability row) plus a one-line plain-English interpretation sentence, per the
  samples in the PDF.
- Do not deviate from the input/output syntax shown — deviations cost marks even if logic is correct.

## Deliverables checklist

- [ ] `designPS10_<GroupID>.pdf` — solution design, ≤4 pages, includes the alternate modeling approach
      + performance tradeoffs.
- [ ] `<GroupID>_Contribution.xlsx` — columns: Student Registration Number, Name, Percentage of
      contribution out of 100%.
- [ ] `inputPS10.txt` — the input used for testing.
- [ ] `outputPS10.txt` — the generated output for that input.
- [ ] One `.py` file with the full implementation.
- [ ] Zip everything as `<GroupID>_A2_PS10.zip` (Group ID format: `Gxxx`, e.g. `G026`).

## Open design decisions to resolve early (affects both code and design doc)

- Exact discretization scheme (# bins, edges, and how "low/high" in query 4 maps to bin labels).
- Structure: hand-crafted DAG vs. learned structure — pick primary + document the other as "alternate."
- Missing-value strategy.
- Whether `Potability` itself is ever a parent of other nodes or purely a sink (naive-Bayes-style) —
  affects both realism and inference cost.

## Implementation status (resolved decisions)

Implemented in [PS10.py](PS10.py); sample [inputPS10.txt](inputPS10.txt) / generated
[outputPS10.txt](outputPS10.txt) are in place. Run with:

```bash
uv run PS10.py
```

- **Discretization**: 3 states (Low/Medium/High) per continuous feature via `pd.qcut` tertiles fit on
  the (median-imputed) training data; outer bin edges widened to ±inf so query-time values outside the
  training range still map to the nearest state instead of erroring.
- **Missing values**: per-column median imputation, fit and applied before discretization.
- **Structure — primary ("expert")**: naive-Bayes-shaped DAG, `Potability -> each feature`. Chosen
  because it (a) matches the causal intuition that potability is what these readings are measuring
  toward, (b) keeps every CPD to one parent so BDeu-smoothed estimates stay stable with ~3.3k rows, and
  (c) lets Variable Elimination marginalize over whatever evidence subset a query omits — required since
  the three sample queries each supply different evidence columns.
- **Structure — alternate**: `--structure learned` runs a Hill-Climb/BIC structure search
  (`pgmpy.estimators.HillClimbSearch` + `BIC`) instead of assuming the naive-Bayes shape. Use this to
  produce the comparison numbers for the design doc's "alternate approach" section (accuracy vs.
  interpretability vs. runtime — the learned search took ~2s on this dataset size, so runtime isn't the
  binding constraint here; the real trade-off to write up is structural: a learned graph can capture
  feature-feature dependencies the expert DAG assumes away, at the cost of a structure that's harder to
  justify/explain and more sensitive to the discretization scheme).
- **CPD estimation**: `DiscreteBayesianEstimator` with a BDeu prior (`equivalent_sample_size=5`) rather
  than raw MLE, so unseen state combinations don't produce zero/undefined probabilities.
- **`Potability`-as-evidence handling**: query 2's sample input includes a `Potability` value alongside
  the other attributes — confirmed as a *reference label only*. `normalize_evidence()` strips it out of
  what's passed to Variable Elimination and echoes it back in the output as
  `(Potability=<x> was given in the input as a reference label; not used as evidence)`.
- **Input file format** (own design choice, since the PDF only shows table renderings, not a
  machine-readable syntax): repeating two-line blocks — comma-separated header naming any subset/order
  of the 9 attributes (+ optional `Potability`), then a comma-separated value line. Values may be raw
  numbers **or** `Low`/`Medium`/`High` labels directly, which is how the qualitative query (low pH, high
  everything else) is expressed without inventing a separate natural-language parser. Blank lines
  separate blocks; `#` lines are comments. Document this format explicitly in the design doc since it's
  not dictated by the assignment spec.
- **Verified**: no-evidence query reproduces the empirical `Potability` base rate (~60.97%/39.03% vs.
  fitted prior 60.97%/39.03%, difference is BDeu smoothing); malformed input (column-count mismatch,
  unknown attribute name, missing CSV) all raise the intended errors with clear messages and correct
  process exit codes.
- **Not yet done**: design doc PDF, contribution spreadsheet, final decision on whether to commit
  `pyproject.toml`/`uv.lock` in the submission zip (see Environment section above).
