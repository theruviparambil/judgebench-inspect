# Changelog

## 0.2.0 (2026-09-29), task version 1

### Scoring: the last VERDICT line is the verdict

The scorer now takes the **last** `VERDICT:` line in a completion. 0.1.0 used Inspect's `pattern()` scorer, which takes the first regex match. The prompt asks the judge to reason and then end its reply with a verdict, so a judge that named one letter, reconsidered, and closed on the other was scored on the line it had abandoned. The register's automated review flagged this, and it is a real, if rare, failure: taking the first match rewards the draft, not the decision.

The change is recorded as Inspect task `version=1` on all four tasks. The grading hash (`a054759f0aa17eb6`) is unchanged because the hash covers the grading *inputs* (judge prompt, verdict regex, dataset, pinned revision, sample construction) and none of those moved. The task version is what records a change to the scoring *rule*.

**Effect on the published runs.** Rescanning every completion behind the published table, exactly 1 of 2,480 judgments contains two VERDICT lines that disagree: `claude-haiku-4-5` on the claude split, pair `6eea64c3-0413-5fdf-b089-80dd2336a931`, original orientation, wrote `VERDICT: B` and then `VERDICT: A` against target B. It was scored correct under 0.1.0 and is incorrect now. Every other judgment contains at most one VERDICT line or repeats the same letter, so the three other runs are unaffected. The published logs were rescored under the new scorer with `scripts/rescore_logs.py` (no model calls; the model outputs are untouched) and the receipts and README were regenerated from the rescored logs. Only the `claude-haiku-4-5` / claude row moves: accuracy 0.739 -> 0.737, position consistency 0.811 -> 0.807, first-position rate 0.472 -> 0.474, and since the pair is an MMLU-Pro law item that per-source accuracy goes 0.682 -> 0.636. The derived statistics move by rounding at most (the haiku split gap +0.090 -> +0.092, z 3.8 -> 3.9; the luna-haiku gap on claude +0.148 -> +0.150; the worst position-bias z 1.29 -> 1.20, still "within 1.3 standard errors") and no conclusion in the README changes.

### Packaging

- `[project.entry-points.inspect_ai]` registers the package, so `inspect eval judgebench_inspect/judgebench_gpt_positional` resolves from an installed package without a file path.
- `tests/test_e2e.py` runs the positional task end to end through Inspect with a `mockllm/model` judge, over the real dataset, and checks that the last-VERDICT rule holds through the whole pipeline. It needs the dataset (network or a warm Hugging Face cache; no token) and skips with a reason otherwise.
- `scripts/extract_results.py` records which scorer produced each receipt and refuses to combine runs scored by different scorers.

### Documentation

- README states that `stderr` treats the two orientations of a pair as independent judgments, which they are not, so the reported intervals are optimistic. The metric is unchanged so the table stays comparable with 0.1.0.

## 0.1.0 (2026-08-25)

Initial release: JudgeBench for Inspect AI with position-consistency measurement, a grading hash stamped into every log, and a README whose figures are pinned against `out/results.json` by tests. Listed in the [inspect_evals register](https://github.com/UKGovernmentBEIS/inspect_evals) at commit `9bb5f35`.
