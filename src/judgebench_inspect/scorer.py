"""Verdict extraction for JudgeBench.

Inspect's built-in `pattern()` scorer takes the FIRST regex match in a
completion. That is the wrong end for a judge. The prompt asks the model to
reason and then end its reply with a verdict, so a judge that names one letter,
reconsiders, and closes on the other has decided on the second. Scoring the
first line scores the discarded draft.

This scorer keeps `pattern()`'s scoring contract (CORRECT / INCORRECT when a
verdict is named, NOANSWER with no `answer` when none is, the completion as the
explanation) and changes exactly one thing: the last VERDICT line is the answer.
Task version 1 records the change; see CHANGELOG.md for its effect on the
published runs.
"""

import re

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# Tolerant of trailing punctuation and markdown emphasis around the letter, but still
# requires the model to have named a verdict. A reply with no VERDICT line scores as
# incorrect rather than being silently coerced to a guess.
VERDICT_PATTERN = r"VERDICT:\s*\**\s*([AB])\b"


def extract_verdict(completion: str) -> str | None:
    """Return the letter named by the LAST verdict line, uppercased, or None if there is none.

    Case-insensitive, like `pattern()`'s default, so `Verdict: b` still counts.
    """
    matches = re.findall(VERDICT_PATTERN, completion, re.IGNORECASE)
    return matches[-1].upper() if matches else None


@scorer(metrics=[accuracy(), stderr()])
def verdict() -> Scorer:
    """Score the judge's final verdict against the target letter.

    A completion with several VERDICT lines is scored on the last one. A
    completion with none scores NOANSWER and carries no `answer`, which is what
    `verdict_parse_rate` counts.
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        answer = extract_verdict(completion)
        if answer is None:
            return Score(
                value=NOANSWER,
                explanation="Scoring pattern not matched in output: " + completion,
            )
        return Score(
            value=CORRECT if answer == target.text.strip().upper() else INCORRECT,
            answer=answer,
            explanation=completion,
        )

    return score
