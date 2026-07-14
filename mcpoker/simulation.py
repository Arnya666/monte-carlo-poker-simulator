"""Monte-Carlo dealing engine.

The simulation deals hands in memory-bounded chunks so it scales to tens of
millions of hands without allocating one giant array. Every run is driven by an
explicit ``numpy`` random generator, so passing a ``seed`` makes results exactly
reproducible.
"""

import numpy as np

from .evaluation import CATEGORIES, classify

HAND_SIZE = 5
DECK_SIZE = 52


def deal(n, rng):
    """Deal ``n`` distinct 5-card hands as an (n, 5) array of card indices.

    For each hand we draw 52 random keys and keep the five smallest. This is
    equivalent to shuffling the deck and taking the top five cards, but it
    vectorises cleanly across all hands at once.
    """
    keys = rng.random((n, DECK_SIZE))
    return np.argpartition(keys, HAND_SIZE, axis=1)[:, :HAND_SIZE].astype(np.int16)


def simulate(n_hands, seed=None, chunk=200_000):
    """Deal and classify ``n_hands`` hands, returning per-category counts.

    The result is an int64 array aligned with ``CATEGORIES``.
    """
    rng = np.random.default_rng(seed)
    totals = np.zeros(len(CATEGORIES), dtype=np.int64)
    dealt = 0
    while dealt < n_hands:
        size = min(chunk, n_hands - dealt)
        categories = classify(deal(size, rng))
        totals += np.bincount(categories, minlength=len(CATEGORIES))
        dealt += size
    return totals
