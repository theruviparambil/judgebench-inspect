"""Tests for the JudgeBench sample construction and position metrics.

The swap is the part most worth guarding. If `record_to_sample(swapped=True)`
ever stops flipping the target alongside the responses, every positional run
silently measures the wrong thing and still reports a plausible number.
"""

import re

import pytest
from inspect_ai.scorer import SampleScore, Score

from judgebench_inspect.metrics import (
    first_position_rate,
    position_consistency,
    verdict_parse_rate,
)
from judgebench_inspect.task import record_to_sample

RECORD = {
    "pair_id": "pair-1",
    "original_id": 42,
    "source": "mmlu-pro-law",
    "question": "Is a warrantless search valid here?",
    "response_model": "gpt-4o-2024-05-13",
    "response_A": "AAA correct answer",
    "response_B": "BBB wrong answer",
    "label": "A>B",
}


def _block(text: str, name: str) -> str:
    match = re.search(rf"<response_{name}>\n(.*?)\n</response_{name}>", text, re.DOTALL)
    assert match is not None, f"no response_{name} block in prompt"
    return match.group(1)


def test_original_orientation_keeps_dataset_order() -> None:
    sample = record_to_sample(RECORD)
    assert _block(sample.input, "A") == "AAA correct answer"
    assert _block(sample.input, "B") == "BBB wrong answer"
    assert sample.target == "A", "label A>B means response_A is correct"


def test_swap_moves_responses_and_flips_target() -> None:
    sample = record_to_sample(RECORD, swapped=True)
    assert _block(sample.input, "A") == "BBB wrong answer"
    assert _block(sample.input, "B") == "AAA correct answer"
    assert sample.target == "B", "correct content moved to slot B, so target must flip"


def test_swap_flips_target_for_b_labels_too() -> None:
    record = {**RECORD, "label": "B>A"}
    assert record_to_sample(record).target == "B"
    assert record_to_sample(record, swapped=True).target == "A"


def test_orientations_share_a_pair_id_but_differ_in_sample_id() -> None:
    original, swapped = record_to_sample(RECORD), record_to_sample(RECORD, swapped=True)
    assert original.metadata["pair_id"] == swapped.metadata["pair_id"]
    assert original.id != swapped.id, "sample ids must be unique within a run"


def test_unexpected_label_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unexpected label"):
        record_to_sample({**RECORD, "label": "A=B"})


def _score(pair_id: str, orientation: str, answer: str) -> SampleScore:
    return SampleScore(
        score=Score(value="C", answer=answer),
        sample_id=f"{pair_id}:{orientation}",
        sample_metadata={"pair_id": pair_id, "orientation": orientation},
    )


def test_content_driven_judge_scores_perfect_consistency() -> None:
    # Letter flips with the order, so the judge tracked the content both times.
    scores = [_score("p1", "original", "A"), _score("p1", "swapped", "B")]
    assert position_consistency()(scores) == 1.0
    assert first_position_rate()(scores) == 0.5


def test_always_a_judge_scores_zero_consistency() -> None:
    scores = [_score("p1", "original", "A"), _score("p1", "swapped", "A")]
    assert position_consistency()(scores) == 0.0
    assert first_position_rate()(scores) == 1.0


def test_incomplete_pairs_are_excluded_not_counted_as_agreement() -> None:
    # p2 only produced one orientation; it must not inflate or deflate the rate.
    scores = [
        _score("p1", "original", "A"),
        _score("p1", "swapped", "B"),
        _score("p2", "original", "A"),
    ]
    assert position_consistency()(scores) == 1.0


def test_unparseable_verdict_lowers_parse_rate_and_drops_from_pairs() -> None:
    scores = [
        _score("p1", "original", "A"),
        _score("p1", "swapped", ""),
    ]
    assert verdict_parse_rate()(scores) == 0.5
    assert position_consistency()(scores) == 0.0, "no complete pair remains"


def test_metrics_return_zero_on_empty_input_rather_than_dividing_by_zero() -> None:
    assert position_consistency()([]) == 0.0
    assert first_position_rate()([]) == 0.0
    assert verdict_parse_rate()([]) == 0.0
