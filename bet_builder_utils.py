"""Utility functions for Bet Builder analyses."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


def calculate_double_chance_labels(
    home_team: str,
    away_team: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
) -> Tuple[Tuple[str, float], Tuple[str, float], Tuple[str, float]]:
    """Return labelled double-chance probabilities for a match.

    The function produces three labelled probability pairs covering the
    combinations commonly used in double-chance betting:

    * ``home_team or Draw``
    * ``away_team or Draw``
    * ``home_team or away_team``

    Parameters
    ----------
    home_team, away_team:
        Names of the home and away teams.
    home_win_prob, draw_prob, away_win_prob:
        Modelled probabilities for the corresponding match outcomes.

    Returns
    -------
    tuple
        ``((home_draw_label, home_draw_prob),
        (away_draw_label, away_draw_prob),
        (home_away_label, home_away_prob))``
    """

    home_draw_label = f"{home_team} or Draw"
    away_draw_label = f"{away_team} or Draw"
    home_away_label = f"{home_team} or {away_team}"

    return (
        (home_draw_label, home_win_prob + draw_prob),
        (away_draw_label, away_win_prob + draw_prob),
        (home_away_label, home_win_prob + away_win_prob),
    )


def generate_accumulator_df(
    match_list: Sequence[Dict[str, object]],
    outcome_types: Sequence[str] = ("home_draw", "away_draw", "home_away"),
) -> pd.DataFrame:
    """Enumerate accumulator combinations for a list of matches.

    Each match dictionary in ``match_list`` is expected to contain keys for the
    outcome odds (``home_win_odds``, ``draw_odds`` and ``away_win_odds``) and
    double-chance probability entries such as ``'home_draw'``.

    Parameters
    ----------
    match_list:
        Iterable of match dictionaries enriched with double-chance selections.
    outcome_types:
        Sequence of outcome keys to consider for each match.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing the selected outcomes for every accumulator,
        along with cumulative probability, cumulative odds and expected value.
    """

    combinations = list(itertools.product(outcome_types, repeat=len(match_list)))
    results: List[Tuple] = []

    for combo in combinations:
        cum_prob = 1.0
        cum_odds = 1.0
        selections: List[str] = []

        for i, outcome in enumerate(combo):
            label, prob = match_list[i][outcome]  # type: ignore[index]
            cum_prob *= prob
            if outcome == "home_draw":
                cum_odds *= match_list[i]["home_win_odds"]  # type: ignore[index]
            elif outcome == "away_draw":
                cum_odds *= match_list[i]["away_win_odds"]  # type: ignore[index]
            else:
                cum_odds *= match_list[i]["draw_odds"]  # type: ignore[index]
            selections.append(label)

        exp_value = cum_prob * (cum_odds - 1)
        results.append((*selections, cum_prob, cum_odds, exp_value))

    columns = [f"Match {i+1} Selection" for i in range(len(match_list))]
    columns += ["Cumulative Probability", "Cumulative Odds", "Expected Value"]
    return pd.DataFrame(results, columns=columns)


def highlight_best_bets(df: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    """Filter accumulator rows to emphasise high expected value bets.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`generate_accumulator_df`.
    top:
        Maximum number of rows to return.

    Returns
    -------
    :class:`pandas.DataFrame`
        Filtered and sorted DataFrame highlighting promising bets.
    """

    ev_thresh = df["Expected Value"].quantile(0.85)
    prob_min = df["Cumulative Probability"].quantile(0.45)
    prob_max = df["Cumulative Probability"].quantile(0.90)
    odds_min = df["Cumulative Odds"].quantile(0.25)

    filtered = df[
        (df["Expected Value"] >= ev_thresh)
        & (df["Cumulative Probability"] >= prob_min)
        & (df["Cumulative Probability"] <= prob_max)
        & (df["Cumulative Odds"] >= odds_min)
    ]

    return filtered.sort_values("Expected Value", ascending=False).head(top)


__all__ = [
    "calculate_double_chance_labels",
    "generate_accumulator_df",
    "highlight_best_bets",
]
