"""
JudgeBench: A Benchmark for Evaluating LLM-based Judges

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron,
Chenguang Wang, Raluca Ada Popa, Ion Stoica
https://arxiv.org/abs/2410.12784

Each sample gives the judge a question and two candidate responses, exactly one of
which is objectively correct. The judge picks the better response, and the gold label
comes from the underlying benchmark's ground truth rather than from human preference.
That is what separates JudgeBench from preference benchmarks: a judge that rewards
fluent, confident, wrong answers scores badly here even when humans would have liked
the answer.

    inspect eval judgebench_inspect/judgebench_gpt --model openai/gpt-4o-mini
    inspect eval judgebench_inspect/judgebench_claude --model anthropic/claude-haiku-4-5-20251001
"""

from functools import partial
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.scorer import accuracy, grouped, stderr
from inspect_ai.solver import generate

from judgebench_inspect.grading import grading_hash
from judgebench_inspect.metrics import (
    first_position_rate,
    position_consistency,
    verdict_parse_rate,
)
from judgebench_inspect.scorer import verdict

# Bumped when the scoring RULE changes over unchanged grading inputs, which the
# grading hash cannot see. 1: the last VERDICT line is the verdict, not the first.
# The history is in CHANGELOG.md.
TASK_VERSION = 1

DATASET_PATH = "ScalerLab/JudgeBench"

# Pinned so results stay reproducible if the upstream dataset is ever revised.
DATASET_REVISION = "57dd5e0b9817d07f05ec8f45a91b2ce1e310e308"

# The two responses are delimited blocks rather than lettered answer options.
#
# This matters more than it looks. JudgeBench items are drawn from MMLU-Pro, so a
# candidate response frequently contains its own "(A) ... (B) ... (C) ..." option list
# while reasoning about the original question. Rendering the two responses as choices
# "A)" and "B)" puts two competing letter schemes in one prompt, and models answer the
# embedded question instead of judging. Delimiting the responses and asking for a
# distinctly named VERDICT keeps the two apart.
JUDGE_PROMPT = """You are evaluating two candidate responses to the same question. Exactly one of them is correct.

Judge on factual and logical correctness only. Ignore length, tone, formatting, and how confident a response sounds.

<question>
{question}
</question>

<response_A>
{response_a}
</response_A>

<response_B>
{response_b}
</response_B>

Decide which response is correct. Reason briefly if it helps, then end your reply with exactly one line:

VERDICT: A
or
VERDICT: B
"""


def record_to_sample(record: dict[str, Any], swapped: bool = False) -> Sample:
    """Convert a JudgeBench row into a judging sample.

    `label` is "A>B" or "B>A", meaning response_A or response_B is the correct one.
    Response order is left exactly as the dataset ships it so that results stay
    comparable with the paper; position effects are measured separately rather than
    shuffled away here.
    """
    label = record["label"]
    if label not in ("A>B", "B>A"):
        raise ValueError(f"unexpected label {label!r} for pair {record['pair_id']!r}")

    correct_is_a = label == "A>B"
    if swapped:
        # Present the responses in the opposite order. The correct content is
        # unchanged, so the correct LETTER flips.
        first, second = record["response_B"], record["response_A"]
        target = "B" if correct_is_a else "A"
    else:
        first, second = record["response_A"], record["response_B"]
        target = "A" if correct_is_a else "B"

    orientation = "swapped" if swapped else "original"
    return Sample(
        input=JUDGE_PROMPT.format(question=record["question"], response_a=first, response_b=second),
        target=target,
        id=f"{record['pair_id']}:{orientation}",
        metadata={
            "pair_id": record["pair_id"],
            "orientation": orientation,
            "source": record["source"],
            "response_model": record["response_model"],
            "original_id": record["original_id"],
        },
    )


def _load(split: str, swapped: bool) -> list[Sample]:
    return list(
        hf_dataset(
            path=DATASET_PATH,
            split=split,
            revision=DATASET_REVISION,
            sample_fields=partial(record_to_sample, swapped=swapped),
        )
    )


def judgebench_task(split: str) -> Task:
    """JudgeBench for one response-generator split, in the order the dataset ships.

    Args:
        split: "gpt" (350 pairs, responses from gpt-4o-2024-05-13) or
            "claude" (270 pairs).
    """
    return Task(
        dataset=MemoryDataset(_load(split, swapped=False)),
        solver=generate(),
        scorer=verdict(),
        version=TASK_VERSION,
        # Stamped into the log so two receipts can be compared only when the
        # inputs that decide a score were identical. See grading.py.
        metadata={"grading_hash": grading_hash()},
        # Accuracy overall, plus a per-source breakdown. The sources are MMLU-Pro
        # subject subsets and judge accuracy is not uniform across them, so a single
        # headline number hides where a judge is actually failing. The parse rate is
        # reported so a judge ignoring the output format is never read as a judge
        # getting the answers wrong.
        metrics=[
            accuracy(),
            stderr(),
            verdict_parse_rate(),
            grouped(accuracy(), "source", all="samples"),
        ],
    )


def judgebench_positional_task(split: str) -> Task:
    """JudgeBench with every pair presented in both orders.

    Doubles the sample count and adds the two metrics accuracy cannot express:
    whether the judge names the same response when the order flips, and how often
    it reaches for the first slot regardless of content. See metrics.py for why
    accuracy alone is not enough on this dataset.

    Args:
        split: "gpt" or "claude".
    """
    return Task(
        # Interleaved, not concatenated. A `--limit N` run takes the first N samples,
        # so appending all the swapped items after all the originals would hand a
        # limited run zero complete pairs and report position_consistency as 0.0.
        dataset=MemoryDataset(
            [
                sample
                for both in zip(
                    _load(split, swapped=False), _load(split, swapped=True), strict=True
                )
                for sample in both
            ]
        ),
        solver=generate(),
        scorer=verdict(),
        version=TASK_VERSION,
        metadata={"grading_hash": grading_hash()},
        metrics=[
            accuracy(),
            stderr(),
            position_consistency(),
            first_position_rate(),
            verdict_parse_rate(),
            grouped(accuracy(), "source", all="samples"),
        ],
    )


@task
def judgebench_gpt() -> Task:
    """JudgeBench over responses generated by gpt-4o-2024-05-13 (350 pairs)."""
    return judgebench_task("gpt")


@task
def judgebench_claude() -> Task:
    """JudgeBench over responses generated by Claude (270 pairs)."""
    return judgebench_task("claude")


@task
def judgebench_gpt_positional() -> Task:
    """JudgeBench gpt split, both orders (700 judgments over 350 pairs)."""
    return judgebench_positional_task("gpt")


@task
def judgebench_claude_positional() -> Task:
    """JudgeBench claude split, both orders (540 judgments over 270 pairs)."""
    return judgebench_positional_task("claude")
