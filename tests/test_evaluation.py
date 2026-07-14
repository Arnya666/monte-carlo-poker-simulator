"""Correctness tests for the hand evaluator and simulator.

The exhaustive test is the strongest possible check: it evaluates every one of
the 2,598,960 distinct 5-card hands and confirms the category counts match the
known combinatorial values exactly. If the evaluator were wrong for any hand
type, these totals would not line up.
"""

import itertools

import numpy as np

from mcpoker.evaluation import CATEGORIES, EXACT_COUNTS, TOTAL_HANDS, classify
from mcpoker.simulation import simulate


def all_five_card_hands():
    """Every distinct 5-card hand as a (2598960, 5) array of card indices."""
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(52), 5)),
        dtype=np.int16,
    )
    return flat.reshape(-1, 5)


def test_exhaustive_counts_match_combinatorics():
    hands = all_five_card_hands()
    assert len(hands) == TOTAL_HANDS

    counts = np.zeros(len(CATEGORIES), dtype=np.int64)
    for start in range(0, len(hands), 300_000):
        block = hands[start:start + 300_000]
        counts += np.bincount(classify(block), minlength=len(CATEGORIES))

    assert dict(zip(CATEGORIES, counts.tolist())) == EXACT_COUNTS


def test_simulation_is_reproducible():
    assert np.array_equal(simulate(50_000, seed=123), simulate(50_000, seed=123))


def test_simulation_counts_sum_to_hand_total():
    counts = simulate(50_000, seed=7)
    assert counts.sum() == 50_000
