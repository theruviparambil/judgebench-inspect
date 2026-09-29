"""End to end: the real dataset, task, scorer and metrics run through Inspect and finish.

A mock judge stands in for the model, so this costs nothing and needs no API
key. Its reply is the same for every sample and names one verdict, reconsiders,
and ends on the other, which makes the run prove three things at once: the
pipeline completes in whatever environment the lockfile installs, the
last-VERDICT rule survives the trip through Inspect, and the swap invariant
holds on the real data (every pair has exactly one orientation whose target is
A, so an always-A judge scores exactly 0.5).

The dataset comes from the Hugging Face Hub (public, no token). With no network
and no warm cache the load raises ConnectionError, and the test skips with that
reason rather than failing on a missing dependency it cannot fix.
"""

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from judgebench_inspect import judgebench_gpt_positional
from judgebench_inspect.task import TASK_VERSION

LIMIT = 4  # two pairs, each in both orientations (the dataset is interleaved)

REPLY = (
    "Response B handles the edge case, so B looks right.\n"
    "VERDICT: B\n\n"
    "Checking again: B drops a term in the last step, so A is the correct one.\n"
    "VERDICT: A\n"
)


def test_positional_task_runs_end_to_end_with_a_mock_judge(tmp_path) -> None:
    try:
        task = judgebench_gpt_positional()
    except ConnectionError as exc:
        pytest.skip(f"JudgeBench dataset unavailable (needs network or a warm HF cache): {exc}")

    judge = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", REPLY)] * LIMIT,
    )
    [log] = eval(task, model=judge, limit=LIMIT, log_dir=str(tmp_path), display="none")

    assert log.status == "success", log.error
    assert log.results is not None
    assert log.results.completed_samples == LIMIT
    assert log.eval.task_version == TASK_VERSION

    [score] = log.results.scores
    assert score.scorer == "verdict"
    metrics = {name: m.value for name, m in score.metrics.items()}
    assert metrics["verdict_parse_rate"] == 1.0
    # Every reply ended on A. If the first VERDICT line had won, this would be 0.0.
    assert metrics["first_position_rate"] == 1.0
    # An always-A judge is position-locked by definition.
    assert metrics["position_consistency"] == 0.0
    # Each pair is correct in exactly one orientation, whichever letter the judge fixes on.
    assert metrics["accuracy"] == 0.5
