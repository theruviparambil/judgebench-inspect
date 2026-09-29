"""A drift guard over the inputs that decide pass or fail.

Model runs cost money, so the expensive failure is not a bug during a run. It is
discovering afterwards that the harness changed between two runs and the numbers
were never comparable. This digests everything that determines a score, stamps it
into every eval log, and pins it with a test, so a change to grading announces
itself as a red test rather than as a confusing result three runs later.

Covered: the judge prompt, the verdict pattern, the dataset path and pinned
revision, and the logic of `record_to_sample` with comments and docstrings
stripped, so prose edits do not churn the hash but a changed rule does.

NOT covered: the scorer and metric implementations themselves, model choice,
generation settings, or anything in Inspect. This is a drift detector for the
things that actually change between runs in this repo, not a proof that two runs
are equivalent. Treat a matching hash as "grading inputs unchanged", nothing more.
A change to the scoring RULE over unchanged inputs is recorded by the task
version (`TASK_VERSION` in task.py) and explained in CHANGELOG.md.
"""

import ast
import hashlib
import inspect
from collections.abc import Callable
from typing import Any


def _normalized_source(fn: Callable[..., Any]) -> str:
    """Return the function's source with comments and docstrings removed.

    Reformatting a comment or rewording a docstring should not invalidate a run
    that graded identically. Changing a branch should.
    """
    tree = ast.parse(inspect.getsource(fn).lstrip())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(ast.fix_missing_locations(tree), annotate_fields=False)


def grading_hash() -> str:
    """Digest of every input that decides a JudgeBench score.

    Imported lazily to avoid a circular import with the task module.
    """
    from judgebench_inspect.scorer import VERDICT_PATTERN
    from judgebench_inspect.task import (
        DATASET_PATH,
        DATASET_REVISION,
        JUDGE_PROMPT,
        record_to_sample,
    )

    parts = [
        ("judge_prompt", JUDGE_PROMPT),
        ("verdict_pattern", VERDICT_PATTERN),
        ("dataset_path", DATASET_PATH),
        ("dataset_revision", DATASET_REVISION),
        ("record_to_sample", _normalized_source(record_to_sample)),
    ]
    digest = hashlib.sha256()
    for name, value in parts:
        # Length-prefixed so two fields cannot be concatenated into a collision.
        digest.update(f"{name}:{len(value)}:".encode())
        digest.update(value.encode())
    return digest.hexdigest()[:16]
