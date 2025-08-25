"""Example scenarios for ``highlight_best_bets``.

This script creates a toy accumulator data set and shows how different
risk parameters affect the resulting selections.
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running the example directly without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from highlight_best_bets import highlight_best_bets

def build_demo_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Expected Value": [0.05, 0.12, -0.02, 0.08, 0.15],
            "Cumulative Probability": [0.55, 0.40, 0.65, 0.30, 0.50],
            "Cumulative Odds": [2.1, 1.8, 2.5, 3.0, 1.6],
        }
    )

def main() -> None:
    df = build_demo_df()

    # Conservative strategy: only consider high probability and high EV bets
    conservative = highlight_best_bets(
        df,
        min_ev=0.1,
        min_prob=0.5,
        max_prob=0.7,
        min_odds=1.5,
        top=3,
    )

    # Aggressive strategy: accept lower probabilities and EVs
    aggressive = highlight_best_bets(
        df,
        min_ev=-0.05,
        min_prob=0.3,
        max_prob=0.8,
        min_odds=1.2,
        top=3,
    )

    print("Conservative strategy selections:\n", conservative, sep="")
    print("\nAggressive strategy selections:\n", aggressive, sep="")

if __name__ == "__main__":
    main()
