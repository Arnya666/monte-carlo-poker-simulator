"""Vectorised evaluation of 5-card poker hands.

Cards are encoded as integers 0-51. The rank is ``card // 4`` (0 = deuce,
12 = ace) and the suit is ``card % 4``. Working with plain integer arrays lets
us classify millions of hands at once with NumPy instead of looping in Python.
"""

import numpy as np

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["c", "d", "h", "s"]

# Hand categories ordered from weakest to strongest. The index into this list is
# the category code returned by ``classify``.
CATEGORIES = [
    "High card",
    "One pair",
    "Two pair",
    "Three of a kind",
    "Straight",
    "Flush",
    "Full house",
    "Four of a kind",
    "Straight flush",
]

# Exact number of distinct 5-card combinations in each category, out of
# C(52, 5) = 2,598,960. "Straight flush" here includes the four royal flushes.
# These are the closed-form combinatorial counts and are what the Monte-Carlo
# frequencies are validated against.
EXACT_COUNTS = {
    "High card": 1_302_540,
    "One pair": 1_098_240,
    "Two pair": 123_552,
    "Three of a kind": 54_912,
    "Straight": 10_200,
    "Flush": 5_108,
    "Full house": 3_744,
    "Four of a kind": 624,
    "Straight flush": 40,
}

TOTAL_HANDS = 2_598_960  # C(52, 5)


def ranks_and_suits(cards):
    """Split card indices (0-51) into (rank, suit) arrays of the same shape."""
    cards = np.asarray(cards)
    return cards // 4, cards % 4


def classify(cards):
    """Classify each 5-card hand into a category code (0-8).

    ``cards`` is an integer array of shape (n, 5) holding card indices 0-51.
    Returns an int8 array of shape (n,) that indexes ``CATEGORIES``.
    """
    cards = np.asarray(cards)
    ranks, suits = ranks_and_suits(cards)

    # Per-hand rank histogram: counts[i, r] is how many cards of rank r hand i
    # holds. The "shape" of these counts distinguishes pairs, trips, quads, etc.
    counts = (ranks[:, :, None] == np.arange(13)).sum(axis=1)  # (n, 13)

    # Multiplicities sorted high-to-low, e.g. a full house is [3, 2, 0, ...].
    shape = np.sort(counts, axis=1)[:, ::-1]
    top, second = shape[:, 0], shape[:, 1]

    # A flush is five cards of one suit.
    flush = (suits == suits[:, :1]).all(axis=1)

    # A straight needs five distinct ranks that are consecutive. The ace-low
    # "wheel" (A-2-3-4-5) is the one straight not caught by max - min == 4.
    present = counts > 0
    distinct = present.sum(axis=1) == 5
    hi = ranks.max(axis=1)
    lo = ranks.min(axis=1)
    normal_straight = distinct & (hi - lo == 4)
    wheel = distinct & present[:, [0, 1, 2, 3, 12]].all(axis=1)
    straight = normal_straight | wheel

    # Assign categories from weakest to strongest so stronger ones overwrite.
    # Straights and flushes only occur among hands with all-distinct ranks, so
    # they never collide with the pair/trips/quad patterns.
    cat = np.zeros(cards.shape[0], dtype=np.int8)
    cat[top == 2] = 1                     # one pair
    cat[(top == 2) & (second == 2)] = 2   # two pair
    cat[top == 3] = 3                     # three of a kind
    cat[straight] = 4
    cat[flush] = 5
    cat[(top == 3) & (second == 2)] = 6   # full house
    cat[top == 4] = 7                     # four of a kind
    cat[straight & flush] = 8             # straight flush (incl. royal)
    return cat
