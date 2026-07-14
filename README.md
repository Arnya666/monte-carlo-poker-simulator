# Monte-Carlo Poker Hand Simulator

A small, fast Monte-Carlo engine for analysing a card game by simulating millions
of hands and turning the results into game economics: hand frequencies,
return-to-player (RTP) and house edge, for any editable paytable.

This demo uses standard 5-card poker as the game, because its exact hand
probabilities are known in closed form. That lets the simulator be **validated
against ground truth**: the same engine and validation approach carry over to a
custom or proprietary game where those closed-form numbers aren't available.

## What it does

- Deals and evaluates hands in vectorised NumPy at around 800,000 hands per second
  on a laptop, scaling to tens of millions in memory-bounded chunks.
- Classifies every hand into the nine standard categories (high card through
  straight flush) with a branch-free evaluator.
- Validates the output three ways: per-category z-scores against exact
  probabilities, a chi-square goodness-of-fit test, and a convergence study.
- Computes RTP and house edge from a JSON paytable that can be edited and
  re-evaluated without re-running any hands.
- Generates a self-contained, theme-aware HTML report (charts as inline SVG, no
  plotting dependencies) that opens in any browser.
- Ships with an exhaustive unit test that checks all 2,598,960 possible hands.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py                    # 1,000,000 hands with the example paytable
python main.py -n 10000000        # 10,000,000 hands
python main.py --convergence      # add a convergence study
python main.py --paytable paytables/example.json
```

For a visual report that opens in a browser (light and dark theme aware):

```bash
python report.py                  # writes report.html for 2,000,000 hands
python report.py -n 5000000 -o out.html
```

Every run is seeded (`--seed`), so results are exactly reproducible.

## How accuracy is validated

The evaluator is checked exhaustively: every possible 5-card hand is classified
and the category totals are compared to the known combinatorial counts (e.g.
exactly 1,098,240 one-pair hands). If any hand type were misclassified, the
totals would not match.

On top of that, each simulation run reports:

- **z-scores**: each simulated frequency should land within a few standard
  errors of its exact probability, where `SE = sqrt(p(1-p)/N)`.
- **Chi-square test**: a single p-value for whether the whole simulated
  distribution is consistent with theory.
- **Convergence**: the maximum error falls off like `1/sqrt(N)` as the number
  of hands grows, the expected Monte-Carlo rate.

```bash
pytest            # runs the exhaustive and reproducibility tests
```

## Layout

```
mcpoker/
  evaluation.py   # card encoding + vectorised hand classifier + exact counts
  simulation.py   # chunked, seedable Monte-Carlo dealing engine
  paytable.py     # RTP / house edge from an editable JSON paytable
validate.py       # z-scores, chi-square test, convergence study
main.py           # command-line report
report.py         # self-contained HTML report with inline-SVG charts
paytables/        # editable paytable definitions
tests/            # exhaustive correctness + reproducibility tests
```

## Adapting to another game

The pieces are deliberately separable. To model a different card game, replace
the hand-evaluation logic in `mcpoker/evaluation.py` and, if needed, the dealing
logic in `mcpoker/simulation.py`; the validation, paytable and reporting layers
stay the same. Where the new game has a smaller state space, exhaustive
enumeration can validate the Monte-Carlo results the same way it does here.
