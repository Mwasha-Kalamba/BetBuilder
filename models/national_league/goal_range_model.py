"""Model to estimate goal count probabilities for National League matches.

This module fits a Poisson regression model using simple pre-match
features and outputs the probability distribution over goal counts for
hold-out matches.  Probabilities are also aggregated into goal ranges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


@dataclass
class GoalRangePrediction:
    """Container for goal count probabilities and aggregates."""
    probabilities: pd.DataFrame
    calibration: pd.DataFrame


def load_match_data(csv_paths: Sequence[str]) -> pd.DataFrame:
    """Load match data and engineer features.

    Parameters
    ----------
    csv_paths:
        Paths to CSV files containing match statistics.

    Returns
    -------
    pd.DataFrame
        Data frame with target total goals and engineered features.
    """
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # Engineer basic features
    data = data.dropna(subset=[
        "home_team_goal_count",
        "away_team_goal_count",
        "team_a_xg",
        "team_b_xg",
        "home_ppg",
        "away_ppg",
    ])

    data["total_goals"] = data["home_team_goal_count"] + data["away_team_goal_count"]
    data["xg_diff"] = data["team_a_xg"] - data["team_b_xg"]
    data["form_diff"] = data["home_ppg"] - data["away_ppg"]

    return data[["total_goals", "xg_diff", "form_diff"]]


def fit_poisson_model(df: pd.DataFrame) -> sm.GLM:
    """Fit a Poisson regression model for total goals."""
    X = sm.add_constant(df[["xg_diff", "form_diff"]])
    y = df["total_goals"]
    model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    return model


def predict_goal_ranges(model: sm.GLM, df: pd.DataFrame, max_goals: int = 8) -> GoalRangePrediction:
    """Generate goal count probabilities and calibration information.

    Parameters
    ----------
    model: sm.GLM
        Fitted Poisson regression model.
    df: pd.DataFrame
        Data frame with features ``xg_diff`` and ``form_diff`` plus
        ``total_goals`` for evaluating calibration.
    max_goals: int, optional
        Maximum number of goals to include in the discrete distribution.

    Returns
    -------
    GoalRangePrediction
        Object containing probability table and calibration results.
    """
    X = sm.add_constant(df[["xg_diff", "form_diff"]])
    mu = model.predict(X)

    # Probabilities for exact goal counts
    goal_range = np.arange(0, max_goals + 1)
    prob_matrix = poisson.pmf(goal_range[:, None], mu).T
    prob_df = pd.DataFrame(prob_matrix, columns=[f"p_{g}_goals" for g in goal_range])

    # Aggregate ranges
    prob_df["p_0_2_goals"] = prob_df[["p_0_goals", "p_1_goals", "p_2_goals"]].sum(axis=1)
    prob_df["p_3_4_goals"] = prob_df[["p_3_goals", "p_4_goals"]].sum(axis=1)
    prob_df["p_5plus_goals"] = 1 - prob_df[["p_0_2_goals", "p_3_4_goals"]].sum(axis=1)

    # Calibration on hold-out matches
    calibration_rows = []
    y_true = df["total_goals"].to_numpy()
    for g in goal_range:
        calibration_rows.append(
            {
                "goals": g,
                "predicted_prob_mean": prob_df[f"p_{g}_goals"].mean(),
                "actual_freq": np.mean(y_true == g),
            }
        )
    calibration_df = pd.DataFrame(calibration_rows)

    return GoalRangePrediction(probabilities=prob_df, calibration=calibration_df)


def train_and_validate(csv_paths: Sequence[str], test_size: float = 0.25, random_state: int = 42) -> GoalRangePrediction:
    """Fit the model on training data and validate on hold-out matches."""
    data = load_match_data(csv_paths)
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    model = fit_poisson_model(train_df)
    prediction = predict_goal_ranges(model, test_df)
    return prediction


if __name__ == "__main__":
    paths = [
        "england-national-league-matches-2023-to-2024-stats.csv",
        "england-national-league-matches-2024-to-2025-stats.csv",
        "england-national-league-matches-2025-to-2026-stats.csv",
    ]
    result = train_and_validate(paths)
    print("Calibration table:")
    print(result.calibration)
