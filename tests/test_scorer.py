"""The verdict scorer reads the judge's final answer, not its first draft.

The published 0.1.0 runs used Inspect's `pattern()` scorer, which takes the
first regex match. The prompt asks for reasoning followed by a closing verdict,
so a reply that names one letter, reconsiders, and ends on the other was being
scored on the line the judge abandoned. These tests pin the new rule and the
one behavior that must not change alongside it: a reply with no verdict is
NOANSWER with no `answer`, so the parse rate still sees it.
"""

import asyncio
import re

from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target
from inspect_ai.solver import TaskState

from judgebench_inspect.scorer import VERDICT_PATTERN, extract_verdict, verdict

TWO_DISAGREEING_VERDICTS = (
    "Response B handles the edge case, so B looks right.\n"
    "VERDICT: B\n\n"
    "Checking the arithmetic again: B drops a term in the last step, A is correct.\n"
    "VERDICT: A\n"
)


def _score(completion: str, target: str) -> Score:
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s1",
        epoch=0,
        input="question",
        messages=[],
        output=ModelOutput.from_content("mockllm/model", completion),
    )
    return asyncio.run(verdict()(state, Target(target)))


def test_last_verdict_line_wins_when_two_disagree() -> None:
    # The first match is the draft the judge walked back; the old scorer took it.
    assert re.search(VERDICT_PATTERN, TWO_DISAGREEING_VERDICTS).group(1) == "B"
    assert extract_verdict(TWO_DISAGREEING_VERDICTS) == "A"

    score = _score(TWO_DISAGREEING_VERDICTS, target="A")
    assert score.answer == "A"
    assert score.value == CORRECT

    score = _score(TWO_DISAGREEING_VERDICTS, target="B")
    assert score.answer == "A"
    assert score.value == INCORRECT


def test_single_verdict_scores_as_before() -> None:
    assert _score("Clearly B.\nVERDICT: B\n", target="B").value == CORRECT
    assert _score("Clearly B.\nVERDICT: B\n", target="A").value == INCORRECT


def test_repeated_agreeing_verdicts_are_not_a_change_of_mind() -> None:
    assert extract_verdict("VERDICT: A\nTo restate: VERDICT: A") == "A"


def test_markdown_emphasis_and_case_are_tolerated() -> None:
    assert extract_verdict("**Verdict: b**") == "B"
    assert extract_verdict("VERDICT: **A**.") == "A"


def test_no_verdict_is_noanswer_and_carries_no_answer() -> None:
    # `verdict_parse_rate` counts samples with no `answer`; a coerced guess
    # would make a judge ignoring the format look like a judge getting it wrong.
    score = _score("Both responses have merit and I cannot decide.", target="A")
    assert score.value == NOANSWER
    assert score.answer is None


def test_letter_must_stand_alone() -> None:
    # "VERDICT: Absolutely" must not parse as A.
    assert extract_verdict("VERDICT: Absolutely B") is None
