"""Position-consistency metrics for JudgeBench.

Accuracy alone cannot separate a judge that reads the responses from one that
prefers whichever response it sees first. JudgeBench's labels lean toward A
(193/350 on the gpt split, 143/270 on claude), so a judge that always answers A
scores 55.1% and 53.0% respectively without reading anything.

The positional tasks run every pair twice, once as shipped and once with the two
responses swapped. That turns position sensitivity into something measurable:

    content-consistent    the judge names the same response both times, which
                          means the LETTER flips when the order flips
    position-locked       the judge names the same LETTER both times, so its
                          answer is determined by slot rather than by content

A judge answering purely on content scores 1.0 on `position_consistency` and
exactly 0.5 on `first_position_rate`. A judge that always answers A scores 0.0
and 1.0.
"""

from collections import defaultdict

from inspect_ai.scorer import Metric, SampleScore, Value, metric


def _verdicts_by_pair(scores: list[SampleScore]) -> dict[str, dict[str, str]]:
    """Group parsed verdict letters by pair_id and orientation.

    Samples whose verdict did not parse carry an empty answer and are dropped, so
    an unparseable reply degrades coverage rather than silently counting as
    agreement. Pairs missing either orientation are dropped by the callers.
    """
    pairs: dict[str, dict[str, str]] = defaultdict(dict)
    for sample_score in scores:
        meta = sample_score.sample_metadata or {}
        pair_id = meta.get("pair_id")
        orientation = meta.get("orientation")
        answer = (sample_score.score.answer or "").strip().upper()
        if pair_id is None or orientation is None or answer not in ("A", "B"):
            continue
        pairs[pair_id][orientation] = answer
    return pairs


@metric
def position_consistency() -> Metric:
    """Fraction of complete pairs where the judge chose the same response both ways.

    Swapping the two responses should flip which letter is correct, so a judge
    reading the content should flip its letter too. Returns 0.0 when no pair has
    both orientations scored.
    """

    def compute(scores: list[SampleScore]) -> Value:
        pairs = _verdicts_by_pair(scores)
        complete = [v for v in pairs.values() if "original" in v and "swapped" in v]
        if not complete:
            return 0.0
        consistent = sum(1 for v in complete if v["original"] != v["swapped"])
        return consistent / len(complete)

    return compute


@metric
def first_position_rate() -> Metric:
    """Fraction of all parsed verdicts naming A, across both orientations.

    Because every pair is presented in both orders, a judge deciding on content
    alone answers A exactly half the time. Values far from 0.5 measure a pull
    toward a slot rather than toward an answer. Returns 0.0 when nothing parsed.
    """

    def compute(scores: list[SampleScore]) -> Value:
        answers = [
            (s.score.answer or "").strip().upper()
            for s in scores
            if (s.sample_metadata or {}).get("orientation") is not None
        ]
        answers = [a for a in answers if a in ("A", "B")]
        if not answers:
            return 0.0
        return sum(1 for a in answers if a == "A") / len(answers)

    return compute


@metric
def verdict_parse_rate() -> Metric:
    """Fraction of judgments that produced a usable VERDICT line.

    Reported so a low accuracy caused by a judge ignoring the output format is
    never mistaken for a judge getting the answers wrong.
    """

    def compute(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        parsed = sum(1 for s in scores if (s.score.answer or "").strip().upper() in ("A", "B"))
        return parsed / len(scores)

    return compute
