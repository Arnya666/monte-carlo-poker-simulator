"""Turn hand frequencies into game economics: return-to-player and house edge.

A paytable maps each hand category to the total amount returned per unit staked
(0 = losing hand, 1 = push, >1 = win). Keeping this separate from the simulation
means a paytable can be edited and re-evaluated without re-running any hands.
"""

import json

from .evaluation import CATEGORIES, EXACT_COUNTS, TOTAL_HANDS


def exact_probabilities():
    """Exact probability of each category as a dict aligned with CATEGORIES."""
    return {c: EXACT_COUNTS[c] / TOTAL_HANDS for c in CATEGORIES}


def load_paytable(path):
    """Load a paytable from JSON and check every category has a payout."""
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    missing = [c for c in CATEGORIES if c not in table]
    if missing:
        raise ValueError(f"Paytable is missing payouts for: {missing}")
    return table


def expected_return(probabilities, paytable):
    """Return-to-player (RTP): expected payout per unit staked."""
    return sum(probabilities[c] * paytable[c] for c in CATEGORIES)


def house_edge(probabilities, paytable):
    """House edge: the fraction of each stake the house keeps on average."""
    return 1.0 - expected_return(probabilities, paytable)
