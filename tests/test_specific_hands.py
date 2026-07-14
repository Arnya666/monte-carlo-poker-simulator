"""Per-hand correctness tests.

The exhaustive test in test_evaluation.py proves the *category totals* match
combinatorics. These tests pin down specific, hand-picked cases so we know the
classifier is right hand-by-hand, including the tricky edge cases (the ace-low
wheel, ace-high Broadway, and near-straights that wrap around the ace and must
NOT count as straights).
"""

import numpy as np
import pytest

from mcpoker.evaluation import CATEGORIES, classify

# Rank codes: 0..12 map to 2,3,4,5,6,7,8,9,T,J,Q,K,A.
T, J, Q, K, A = 8, 9, 10, 11, 12


def card(rank, suit):
    """Encode a (rank 0-12, suit 0-3) pair as a card index 0-51."""
    return rank * 4 + suit


# Each case: a readable hand and the category we expect it to classify as.
CASES = {
    "Royal flush": ([(T, 0), (J, 0), (Q, 0), (K, 0), (A, 0)], "Straight flush"),
    "Straight flush 5-9": ([(3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], "Straight flush"),
    "Wheel straight flush": ([(A, 2), (0, 2), (1, 2), (2, 2), (3, 2)], "Straight flush"),
    "Four of a kind": ([(A, 0), (A, 1), (A, 2), (A, 3), (K, 0)], "Four of a kind"),
    "Full house": ([(K, 0), (K, 1), (K, 2), (Q, 0), (Q, 1)], "Full house"),
    "Flush": ([(0, 0), (3, 0), (5, 0), (7, 0), (A, 0)], "Flush"),
    "Straight (mixed suits)": ([(3, 0), (4, 1), (5, 2), (6, 3), (7, 0)], "Straight"),
    "Broadway straight": ([(T, 0), (J, 1), (Q, 2), (K, 3), (A, 0)], "Straight"),
    "Wheel straight": ([(A, 0), (0, 1), (1, 2), (2, 3), (3, 0)], "Straight"),
    "Three of a kind": ([(5, 0), (5, 1), (5, 2), (0, 0), (7, 0)], "Three of a kind"),
    "Two pair": ([(6, 0), (6, 1), (1, 0), (1, 1), (A, 0)], "Two pair"),
    "One pair": ([(3, 0), (3, 1), (0, 0), (5, 0), (7, 0)], "One pair"),
    "High card": ([(0, 0), (3, 1), (5, 2), (7, 3), (K, 0)], "High card"),
    # Ace wraps that must NOT be straights:
    "A-2-3-4-6 is not a straight": ([(A, 0), (0, 1), (1, 2), (2, 3), (4, 0)], "High card"),
    "J-Q-K-A-2 is not a straight": ([(J, 0), (Q, 1), (K, 2), (A, 3), (0, 0)], "High card"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_specific_hand(name):
    cards, expected = CASES[name]
    hand = np.array([[card(r, s) for r, s in cards]])
    result = CATEGORIES[classify(hand)[0]]
    assert result == expected, f"{name}: got {result!r}, expected {expected!r}"


def test_all_cards_are_distinct_and_in_range():
    """Guard against typos in the test data itself."""
    for name, (cards, _) in CASES.items():
        indices = [card(r, s) for r, s in cards]
        assert len(set(indices)) == 5, f"{name} has duplicate cards"
        assert all(0 <= c < 52 for c in indices), f"{name} has an out-of-range card"
