"""Monte-Carlo analysis of 5-card poker hands.

Deals a large number of random hands, validates the resulting frequencies
against exact combinatorics, and evaluates a paytable to produce the
return-to-player and house edge.

Usage:
    python main.py                       # 1,000,000 hands, default paytable
    python main.py -n 10000000           # 10,000,000 hands
    python main.py --paytable my.json    # a different paytable
    python main.py --convergence         # also run a convergence study
"""

import argparse

from mcpoker.evaluation import CATEGORIES
from mcpoker.paytable import (exact_probabilities, expected_return,
                              house_edge, load_paytable)
from mcpoker.simulation import simulate
from validate import chi_square, comparison_table, convergence


def print_comparison(counts):
    print(f"\nSimulated {counts.sum():,} hands\n")
    header = f"{'Category':<18}{'Simulated':>12}{'Exact':>12}{'Abs err':>12}{'z':>8}"
    print(header)
    print("-" * len(header))
    for row in comparison_table(counts):
        print(f"{row['category']:<18}"
              f"{row['simulated']:>12.6f}"
              f"{row['exact']:>12.6f}"
              f"{row['abs_error']:>+12.6f}"
              f"{row['z']:>8.2f}")

    chi2, p, dof = chi_square(counts)
    print(f"\nChi-square goodness of fit: chi2={chi2:.2f}, dof={dof}, p={p:.3f}")
    print("(a large p-value means the simulated distribution is consistent "
          "with exact theory)")


def print_paytable(counts, paytable_path):
    paytable = load_paytable(paytable_path)
    p_exact = exact_probabilities()
    p_sim = {c: counts[i] / counts.sum() for i, c in enumerate(CATEGORIES)}

    print(f"\nPaytable: {paytable_path}\n")
    print(f"{'Category':<18}{'Payout':>8}{'P(exact)':>12}")
    print("-" * 38)
    for c in CATEGORIES:
        print(f"{c:<18}{paytable[c]:>8}{p_exact[c]:>12.6f}")

    print(f"\nReturn to player (exact):     {expected_return(p_exact, paytable):>9.4%}")
    print(f"Return to player (simulated): {expected_return(p_sim, paytable):>9.4%}")
    print(f"House edge (exact):           {house_edge(p_exact, paytable):>9.4%}")


def print_convergence(seed):
    print("\nConvergence study (mean max-error over all categories, 12 seeds each)\n")
    print(f"{'Hands':>12}{'Mean error':>14}{'error*sqrt(N)':>16}")
    print("-" * 42)
    for n, err, scaled in convergence([10_000, 50_000, 250_000, 1_250_000],
                                      base_seed=seed):
        print(f"{n:>12,}{err:>14.6f}{scaled:>16.4f}")
    print("\nEach 5x in hands roughly halves the error (1/sqrt(N)); the last")
    print("column stays near-constant, confirming the Monte-Carlo convergence rate.")


def main():
    parser = argparse.ArgumentParser(
        description="Monte-Carlo analysis of 5-card poker hands.")
    parser.add_argument("-n", "--hands", type=int, default=1_000_000,
                        help="number of hands to simulate (default: 1,000,000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducibility (default: 0)")
    parser.add_argument("--paytable", default="paytables/example.json",
                        help="path to a paytable JSON file")
    parser.add_argument("--convergence", action="store_true",
                        help="also run a convergence study")
    args = parser.parse_args()

    counts = simulate(args.hands, seed=args.seed)
    print_comparison(counts)
    print_paytable(counts, args.paytable)
    if args.convergence:
        print_convergence(args.seed)


if __name__ == "__main__":
    main()
