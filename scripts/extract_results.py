"""Extract a compact, committable receipt from the full eval logs.

The `.eval` logs are 6-8 MB each and belong in the register's log store, not in
git. This writes the numbers the README cites into `out/results.json` so
`tests/test_readme_claims.py` can pin every published figure against a receipt
rather than against a number somebody typed.

Every run must carry the same grading hash. A mismatch means the runs were
produced under different rules and must not appear in one table, so this refuses
to write rather than emitting a receipt that invites an invalid comparison.

    uv run python scripts/extract_results.py
"""

import json
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log

LOG_DIR = Path("logs-full")
OUT = Path("out/results.json")

METRICS = (
    "accuracy",
    "stderr",
    "position_consistency",
    "first_position_rate",
    "verdict_parse_rate",
)


def main() -> None:
    runs = []
    hashes = set()

    for info in list_eval_logs(str(LOG_DIR)):
        log = read_eval_log(info.name, header_only=True)
        if log.status != "success":
            continue
        score = log.results.scores[0]
        grading_hash = (log.eval.metadata or {}).get("grading_hash")
        hashes.add(grading_hash)
        runs.append(
            {
                "model": log.eval.model,
                "task": log.eval.task,
                "split": log.eval.task.replace("judgebench_", "").replace("_positional", ""),
                "judgments": log.results.total_samples,
                "pairs": log.results.total_samples // 2,
                "grading_hash": grading_hash,
                "metrics": {
                    name: round(metric.value, 4)
                    for name, metric in score.metrics.items()
                    if name in METRICS
                },
                "per_source": {
                    name: round(metric.value, 4)
                    for name, metric in score.metrics.items()
                    if name.startswith("mmlu-pro")
                },
            }
        )

    if not runs:
        raise SystemExit(f"no successful runs found in {LOG_DIR}/")
    if len(hashes) != 1:
        raise SystemExit(
            f"runs carry different grading hashes {hashes}; they were produced "
            "under different rules and must not be published in one table"
        )

    runs.sort(key=lambda r: (r["model"], r["split"]))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"grading_hash": hashes.pop(), "runs": runs}, indent=2) + "\n")
    print(f"wrote {OUT} with {len(runs)} runs")


if __name__ == "__main__":
    main()
