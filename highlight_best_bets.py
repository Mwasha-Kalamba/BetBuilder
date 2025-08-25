"""Bet filtering utilities.

This module exposes :func:`highlight_best_bets` which filters accumulator
combinations based on expected value, cumulative probability and cumulative
odds.  Users can supply absolute threshold values to tailor the selection to
their risk profile.  When a threshold is not provided, sensible percentile
defaults are used to emulate the previous behaviour of the project.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


def highlight_best_bets(
    df: pd.DataFrame,
    top: int = 50,
    *,
    min_ev: Optional[float] = None,
    min_prob: Optional[float] = None,
    max_prob: Optional[float] = None,
    min_odds: Optional[float] = None,
    ev_quantile: float = 0.85,
    prob_min_quantile: float = 0.45,
    prob_max_quantile: float = 0.90,
    odds_min_quantile: float = 0.25,
) -> pd.DataFrame:
    """Return the strongest accumulator combinations.

    Parameters
    ----------
    df:
        DataFrame with at least ``Expected Value``, ``Cumulative Probability``
        and ``Cumulative Odds`` columns.
    top:
        Maximum number of rows to return after filtering.
    min_ev, min_prob, max_prob, min_odds:
        Absolute thresholds used to filter the DataFrame.  When ``None`` the
        respective ``*_quantile`` value is used instead to maintain the old
        percentile based defaults.
    ev_quantile, prob_min_quantile, prob_max_quantile, odds_min_quantile:
        Quantile defaults used when an absolute threshold is not supplied.

    Returns
    -------
    pandas.DataFrame
        Filtered and sorted by expected value in descending order.
    """

    ev_thresh = (
        df["Expected Value"].quantile(ev_quantile)
        if min_ev is None
        else min_ev
    )
    prob_min_val = (
        df["Cumulative Probability"].quantile(prob_min_quantile)
        if min_prob is None
        else min_prob
    )
    prob_max_val = (
        df["Cumulative Probability"].quantile(prob_max_quantile)
        if max_prob is None
        else max_prob
    )
    odds_min_val = (
        df["Cumulative Odds"].quantile(odds_min_quantile)
        if min_odds is None
        else min_odds
    )

    filtered = df[
        (df["Expected Value"] >= ev_thresh)
        & (df["Cumulative Probability"] >= prob_min_val)
        & (df["Cumulative Probability"] <= prob_max_val)
        & (df["Cumulative Odds"] >= odds_min_val)
    ]

    return filtered.sort_values("Expected Value", ascending=False).head(top)


if __name__ == "__main__":
    # Simple demonstration when module executed directly.
    data = {
        "Expected Value": [0.05, 0.12, -0.02, 0.08, 0.15],
        "Cumulative Probability": [0.55, 0.40, 0.65, 0.30, 0.50],
        "Cumulative Odds": [2.1, 1.8, 2.5, 3.0, 1.6],
    }
    demo_df = pd.DataFrame(data)

    conservative = highlight_best_bets(
        demo_df, min_ev=0.1, min_prob=0.5, max_prob=0.7, min_odds=1.5, top=3
    )
    aggressive = highlight_best_bets(
        demo_df, min_ev=-0.05, min_prob=0.3, max_prob=0.8, min_odds=1.2, top=3
    )

    print("Conservative strategy:\n", conservative)
    print("\nAggressive strategy:\n", aggressive)
