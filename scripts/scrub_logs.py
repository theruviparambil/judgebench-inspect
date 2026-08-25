"""Redact local filesystem paths from eval logs before publishing them.

The inspect_evals register asks submitters to upload two full log files and warns
that "these logs may be made public, as a result you should consider
post-processing to de-identifying them (i.e removing paths with usernames or
other PII)".

Unbatched runs capture Python tracebacks on retries, and those tracebacks embed
absolute paths containing the operating user's home directory. In the runs that
produced this repo's published numbers that was ~4,400 occurrences per log, all
rooted at the submitter's username. Batched runs happened to be clean, which is
luck rather than a property to rely on.

This reads each log, rewrites every absolute home path to a placeholder, and
writes a sanitized copy to `logs-public/`. Originals are left untouched.

    uv run python scripts/scrub_logs.py
    uv run python scripts/scrub_logs.py --check   # verify, write nothing

Verify the output before uploading. A scrubber that silently misses a pattern is
worse than no scrubber, so `--check` re-scans the written files and exits
non-zero if anything identifying survives.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

SRC = Path("logs-full")
DST = Path("logs-public")

HOME = re.compile(r"/(?:Users|home)/[^/\s\"'\\)]+")
PLACEHOLDER = "/home/redacted"

# Matched against the sanitized output. `sk-` followed by enough characters to
# be a real credential; short `sk-` fragments occur inside opaque provider
# reasoning blobs and are not secrets.
LEFTOVER = re.compile(
    r"/(?:Users|home)/(?!redacted)[^/\s\"'\\)]+"
    r"|(?<![A-Za-z0-9])sk-(?:proj-|ant-|or-)?[A-Za-z0-9_-]{20,}"
)


def _scrub(obj: object) -> object:
    if isinstance(obj, str):
        return HOME.sub(PLACEHOLDER, obj)
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    return obj


def _identifying(log_path: Path) -> list[str]:
    log = read_eval_log(str(log_path))
    blob = json.dumps(log.model_dump(), default=str)
    return sorted(set(LEFTOVER.findall(blob)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify only")
    args = parser.parse_args()

    if args.check:
        failures = 0
        for path in sorted(DST.glob("*.eval")):
            found = _identifying(path)
            print(f"{path.name[:52]:54} {'CLEAN' if not found else found[:3]}")
            failures += bool(found)
        if failures:
            print(f"\n{failures} sanitized log(s) still contain identifying strings")
        return 1 if failures else 0

    DST.mkdir(exist_ok=True)
    written = 0
    for info in list_eval_logs(str(SRC)):
        log = read_eval_log(info.name)
        if log.status != "success":
            # Failed runs are not part of a submission and carry the most
            # traceback noise; skip rather than sanitize them.
            continue
        scrubbed = type(log).model_validate(_scrub(log.model_dump()))
        out = DST / Path(info.name).name
        write_eval_log(scrubbed, str(out))
        written += 1
        print(f"wrote {out}")
    print(f"\n{written} log(s) sanitized into {DST}/; now run with --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
