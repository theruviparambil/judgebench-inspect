"""Rescore the full logs under the package's current scorer, without calling any model.

For a change that alters how a completion is scored but nothing that decides
what the model was asked (prompt, dataset, revision, sample construction), the
right rerun is a rescore: the model outputs are the expensive, nondeterministic
part and they are unchanged. A fresh model run would move every number by
sampling noise and hide the one thing that actually changed.

Each original is copied to `logs-prescore/` before it is rewritten, and an
existing copy there is never overwritten, so the pre-rescore logs survive
repeated runs. Metric deltas are printed per log; then regenerate the receipts
and check the README:

    uv run python scripts/rescore_logs.py
    uv run python scripts/extract_results.py
    uv run pytest tests/test_readme_claims.py -q
    uv run python scripts/scrub_logs.py && uv run python scripts/scrub_logs.py --check
"""

import os
import shutil
from pathlib import Path

# Inspect initializes the log's model as the active model for the scoring pass
# even when, as here, the scorer never calls it. The providers refuse to build
# a client without a key, so a placeholder satisfies them. Nothing is sent.
for _var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(_var, "unused-rescore-only")

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from judgebench_inspect.scorer import verdict

LOG_DIR = Path("logs-full")
BACKUP_DIR = Path("logs-prescore")

REPORTED = ("accuracy", "position_consistency", "first_position_rate", "verdict_parse_rate")


def main() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    for path in sorted(LOG_DIR.glob("*.eval")):
        backup = BACKUP_DIR / path.name
        if not backup.exists():
            shutil.copy2(path, backup)

        log = read_eval_log(str(path))
        if log.status != "success" or log.results is None:
            print(f"skip {path.name[:52]} (status {log.status})")
            continue
        before = {k: m.value for k, m in log.results.scores[0].metrics.items()}
        rescored = score(log, scorers=verdict(), action="overwrite", display="none")
        assert rescored.results is not None
        after = {k: m.value for k, m in rescored.results.scores[0].metrics.items()}
        write_eval_log(rescored, str(path))

        print(f"{path.name[:52]}  {log.eval.model}")
        for name in REPORTED:
            flag = "" if before[name] == after[name] else "   <- changed"
            print(f"    {name:22} {before[name]:.4f} -> {after[name]:.4f}{flag}")


if __name__ == "__main__":
    main()
