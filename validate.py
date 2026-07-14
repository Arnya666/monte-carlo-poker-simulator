"""Validation harness: check Monte-Carlo estimates against exact combinatorics.

This is the core evidence that the simulator is statistically accurate:

1. Every category's simulated frequency is compared with its exact probability,
   with the deviation reported in standard errors (a z-score).
2. A Pearson chi-square test checks the whole distribution at once.
3. A convergence study shows the error shrinking as the hand count grows.
"""

import numpy as np
from scipy import stats

from mcpoker.evaluation import CATEGORIES, EXACT_COUNTS, TOTAL_HANDS
from mcpoker.simulation import simulate


def exact_probs():
    """Exact category probabilities as an array aligned with CATEGORIES."""
    return np.array([EXACT_COUNTS[c] for c in CATEGORIES]) / TOTAL_HANDS


def comparison_table(counts):
    """Per-category comparison of simulated vs exact probabilities.

    ``z`` is the deviation measured in standard errors; for an accurate
    simulator it should sit within roughly +/-3 for every category.
    """
    n = counts.sum()
    p_hat = counts / n
    p_exact = exact_probs()
    se = np.sqrt(p_exact * (1 - p_exact) / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p_hat - p_exact) / se, 0.0)

    return [
        {
            "category": c,
            "simulated": p_hat[i],
            "exact": p_exact[i],
            "abs_error": p_hat[i] - p_exact[i],
            "z": z[i],
        }
        for i, c in enumerate(CATEGORIES)
    ]


def chi_square(counts):
    """Pearson chi-square goodness-of-fit against the exact distribution."""
    n = counts.sum()
    expected = exact_probs() * n
    # Guard against tiny floating-point drift so observed and expected totals
    # match exactly, which scipy requires.
    expected *= n / expected.sum()
    chi2, p = stats.chisquare(counts, expected)
    dof = len(CATEGORIES) - 1
    return chi2, p, dof


def convergence(sizes, trials=12, base_seed=0):
    """Mean max-error at each size, averaged over several seeds.

    A single run's max error is noisy, so the trend is measured by averaging
    over ``trials`` independent seeds. The averaged error should fall like
    ``1/sqrt(N)``; equivalently ``error * sqrt(N)`` stays roughly constant.
    """
    p_exact = exact_probs()
    results = []
    for n in sizes:
        errors = []
        for t in range(trials):
            counts = simulate(n, seed=base_seed + t)
            p_hat = counts / counts.sum()
            errors.append(np.abs(p_hat - p_exact).max())
        mean_error = float(np.mean(errors))
        results.append((n, mean_error, mean_error * np.sqrt(n)))
    return results
