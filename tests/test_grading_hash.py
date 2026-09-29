"""Drift guard over the grading hash.

If `test_grading_hash_is_pinned` fails, something that decides a score changed.
That is not automatically a bug, but it does mean **every committed result was
produced under different rules and is no longer comparable**. The fix is to
decide deliberately: either revert the change, or update the pin here, re-run,
and replace the published numbers. Never update the pin alone to make the suite
green, because that is exactly the silent-drift failure this file exists to stop.
"""

from judgebench_inspect.grading import _normalized_source, grading_hash
from judgebench_inspect.task import TASK_VERSION, judgebench_positional_task, judgebench_task

# Bump ONLY together with a re-run and a README update. See the module docstring.
EXPECTED_GRADING_HASH = "a054759f0aa17eb6"

# The hash does not see the scoring rule; the task version records it. Bump ONLY
# together with a CHANGELOG entry and a rescore of logs-full/ (scripts/rescore_logs.py).
EXPECTED_TASK_VERSION = 1


def test_grading_hash_is_pinned() -> None:
    assert grading_hash() == EXPECTED_GRADING_HASH, (
        "grading inputs changed; committed results are no longer comparable. "
        "Re-run and update both the pin and the README, or revert the change."
    )


def test_grading_hash_is_deterministic() -> None:
    assert grading_hash() == grading_hash()


def test_hash_and_version_are_stamped_into_every_task() -> None:
    assert TASK_VERSION == EXPECTED_TASK_VERSION, (
        "task version changed; rescore logs-full/, regenerate the receipts, "
        "and explain the change in CHANGELOG.md before updating this pin"
    )
    for build in (judgebench_task, judgebench_positional_task):
        for split in ("gpt", "claude"):
            built = build(split)
            metadata = built.metadata or {}
            assert metadata.get("grading_hash") == EXPECTED_GRADING_HASH, (
                f"{build.__name__}({split!r}) did not carry the grading hash; "
                "a receipt without it cannot be checked for drift"
            )
            assert built.version == EXPECTED_TASK_VERSION, (
                f"{build.__name__}({split!r}) does not carry the task version"
            )


def test_comments_and_docstrings_do_not_change_the_normalized_source() -> None:
    # Rewording prose must not invalidate a run that graded identically.
    def with_prose(x: int) -> int:
        """A docstring."""
        # a comment
        return x + 1

    def without_prose(x: int) -> int:
        return x + 1

    a = _normalized_source(with_prose).replace("with_prose", "F")
    b = _normalized_source(without_prose).replace("without_prose", "F")
    assert a == b


def test_a_logic_change_does_change_the_normalized_source() -> None:
    # The guard is worthless if it cannot see a changed rule.
    def before(x: int) -> int:
        return x + 1

    def after(x: int) -> int:
        return x + 2

    a = _normalized_source(before).replace("before", "F")
    b = _normalized_source(after).replace("after", "F")
    assert a != b
