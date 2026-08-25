"""Pin every number in the README against the receipts in out/results.json.

A README is the part of a repo people actually read, and it is the part that
silently goes stale. These tests parse the published file and check its figures
against the extracted receipts, so an edited number, a re-run that moved a
result, or a copied-forward figure from an earlier run all fail here instead of
being discovered by a reader.

Deliberately asserts against the RECEIPT, never against a constant re-typed in
this file. A test that restates the number it is checking proves nothing.
"""

import json
import math
import re
from pathlib import Path

import pytest

README = Path("README.md").read_text()
RESULTS = json.loads(Path("out/results.json").read_text())
RUNS = {(r["model"].split("/")[-1], r["split"]): r for r in RESULTS["runs"]}


def _row(model: str, split: str) -> dict:
    return RUNS[(model, split)]


def _z(a: dict, b: dict) -> float:
    diff = a["metrics"]["accuracy"] - b["metrics"]["accuracy"]
    se = math.hypot(a["metrics"]["stderr"], b["metrics"]["stderr"])
    return diff / se


@pytest.mark.parametrize(
    ("model", "split"),
    [
        ("gpt-5.6-luna", "gpt"),
        ("gpt-5.6-luna", "claude"),
        ("claude-haiku-4-5", "gpt"),
        ("claude-haiku-4-5", "claude"),
    ],
)
def test_results_table_row_matches_receipt(model: str, split: str) -> None:
    run = _row(model, split)
    m = run["metrics"]
    # The row is: | `model` | price | split | pairs | acc ±se | consistency | fp | parse |
    pattern = (
        rf"\|\s*`{re.escape(model)}`\s*\|[^|]+\|\s*{split}\s*\|\s*(\d+)\s*\|"
        # \**  absorbs the bold markers around a highlighted accuracy value.
        rf"[^|]*?([\d.]+)\**\s*±\s*([\d.]+)[^|]*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )
    match = re.search(pattern, README)
    assert match, f"no results row found in README for {model} / {split}"
    pairs, acc, se, cons, first, parse = match.groups()

    assert int(pairs) == run["pairs"]
    assert float(acc) == pytest.approx(m["accuracy"], abs=0.001)
    assert float(se) == pytest.approx(m["stderr"], abs=0.001)
    assert float(cons) == pytest.approx(m["position_consistency"], abs=0.001)
    assert float(first) == pytest.approx(m["first_position_rate"], abs=0.001)
    assert float(parse) == pytest.approx(m["verdict_parse_rate"], abs=0.001)


def test_grading_hash_in_readme_matches_receipts() -> None:
    assert RESULTS["grading_hash"] in README, (
        "README does not cite the grading hash its numbers were produced under"
    )


def test_total_judgment_count_matches_receipts() -> None:
    total = sum(r["judgments"] for r in RESULTS["runs"])
    assert f"{total:,} judgments" in README, f"README should state {total:,} judgments"


def test_model_gap_z_scores_match_the_receipts() -> None:
    # README claims +0.079 (z = 4.4) on gpt and +0.148 (z = 6.4) on claude.
    for split, claimed_diff, claimed_z in (("gpt", 0.079, 4.4), ("claude", 0.148, 6.4)):
        luna, haiku = _row("gpt-5.6-luna", split), _row("claude-haiku-4-5", split)
        diff = luna["metrics"]["accuracy"] - haiku["metrics"]["accuracy"]
        assert diff == pytest.approx(claimed_diff, abs=0.001), split
        assert _z(luna, haiku) == pytest.approx(claimed_z, abs=0.1), split
        assert f"+{claimed_diff:.3f} (z = {claimed_z}" in README


def test_self_preference_table_matches_the_receipts() -> None:
    for model, claimed_diff, claimed_z in (
        ("gpt-5.6-luna", 0.020, 1.1),
        ("claude-haiku-4-5", 0.090, 3.8),
    ):
        gpt, claude = _row(model, "gpt"), _row(model, "claude")
        diff = gpt["metrics"]["accuracy"] - claude["metrics"]["accuracy"]
        assert diff == pytest.approx(claimed_diff, abs=0.001), model
        assert _z(gpt, claude) == pytest.approx(claimed_z, abs=0.1), model
        assert f"+{claimed_diff:.3f} | {claimed_z} |" in README


def test_luna_split_gap_is_still_not_significant() -> None:
    # The README says "no detectable difference" for luna across splits. If a
    # re-run makes that gap significant, the prose is wrong and must change.
    z = _z(_row("gpt-5.6-luna", "gpt"), _row("gpt-5.6-luna", "claude"))
    assert abs(z) < 2, (
        f"luna's split gap is now z={z:.1f}; the README claims no detectable "
        "difference and must be rewritten"
    )
    assert "no detectable difference" in README


def test_haiku_split_gap_is_still_significant_and_in_the_stated_direction() -> None:
    # The whole self-preference argument rests on this being negative for a
    # Claude judge on Claude-written responses.
    z = _z(_row("claude-haiku-4-5", "gpt"), _row("claude-haiku-4-5", "claude"))
    assert z >= 2, f"haiku's split gap is now z={z:.1f}; the argument no longer holds"
    assert "opposite" in README


def test_no_run_shows_detectable_position_bias() -> None:
    # README: "all four runs ... every one within 1.3 standard errors".
    worst = 0.0
    for run in RESULTS["runs"]:
        n = run["judgments"]
        z = abs(run["metrics"]["first_position_rate"] - 0.5) / math.sqrt(0.25 / n)
        worst = max(worst, z)
    assert worst < 2, f"a run now shows position bias (z={worst:.1f})"
    assert "within 1.3 standard errors" in README
    assert worst <= 1.3, f"worst position-bias z is {worst:.2f}, README says 1.3"


def test_always_a_baselines_match_the_dataset() -> None:
    # These come from the dataset's label balance, not from a run.
    for split, correct, total, pct in (("gpt", 193, 350, 55.1), ("claude", 143, 270, 53.0)):
        assert round(100 * correct / total, 1) == pct, split
        assert f"{pct}%** ({correct}/{total})" in README


def test_every_run_shares_one_grading_hash() -> None:
    hashes = {r["grading_hash"] for r in RESULTS["runs"]}
    assert hashes == {RESULTS["grading_hash"]}, (
        f"receipts carry mixed grading hashes {hashes}; they are not comparable"
    )
