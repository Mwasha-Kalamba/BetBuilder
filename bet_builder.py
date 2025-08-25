from dataclasses import dataclass
import itertools
import numpy as np
from typing import List, Tuple, Iterable, Optional

@dataclass
class Match:
    """Simple container for match information."""
    name: str
    prob: float  # win probability
    odds: float  # decimal odds


def joint_probability(probs: Iterable[float], corr_matrix: Optional[np.ndarray] = None) -> float:
    """Approximate joint probability with optional pairwise correlation.

    The baseline assumes independence (product of probabilities).  If a
    correlation matrix is supplied, pairwise covariance adjustments are added:

    P(A and B) = pA * pB + rho * sqrt(pA(1-pA) * pB(1-pB))

    The approximation extends this pairwise adjustment to multiple events by
    summing covariances.  The result is clipped to [0, 1].
    """
    probs = list(probs)
    joint = float(np.prod(probs))
    if corr_matrix is not None:
        n = len(probs)
        cov_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                rho = corr_matrix[i, j]
                cov = rho * np.sqrt(probs[i] * (1 - probs[i]) * probs[j] * (1 - probs[j]))
                cov_sum += cov
        joint += cov_sum
        joint = max(min(joint, 1.0), 0.0)
    return joint


def cartesian_parlays(matches: List[Match], max_len: int = 3, corr_matrix: Optional[np.ndarray] = None) -> List[Tuple[Tuple[int, ...], float]]:
    """Evaluate EV for all combinations up to ``max_len`` via exhaustive search."""
    parlays = []
    for r in range(1, max_len + 1):
        for combo in itertools.combinations(range(len(matches)), r):
            probs = [matches[i].prob for i in combo]
            odds = float(np.prod([matches[i].odds for i in combo]))
            sub_corr = corr_matrix[np.ix_(combo, combo)] if corr_matrix is not None else None
            p = joint_probability(probs, sub_corr)
            ev = p * odds - 1
            parlays.append((combo, ev))
    return parlays


def greedy_parlays(
    matches: List[Match],
    ev_threshold: float = 0.05,
    max_len: int = 3,
    corr_matrix: Optional[np.ndarray] = None,
) -> List[Tuple[Tuple[int, ...], float]]:
    """Greedy heuristic to build parlays exceeding ``ev_threshold``.

    Starting from high individual EV legs, additional legs are added only when
    the new parlay still satisfies the EV threshold.  This limits the number of
    evaluated combinations relative to an exhaustive cartesian product.
    """
    # sort matches by individual EV descending
    base_order = sorted(
        range(len(matches)), key=lambda i: matches[i].prob * matches[i].odds - 1, reverse=True
    )

    parlays = []
    for idx in base_order:
        combo = [idx]
        added = True
        while added and len(combo) < max_len:
            added = False
            for j in base_order:
                if j in combo:
                    continue
                candidate = combo + [j]
                probs = [matches[i].prob for i in candidate]
                odds = float(np.prod([matches[i].odds for i in candidate]))
                sub_corr = corr_matrix[np.ix_(candidate, candidate)] if corr_matrix is not None else None
                p = joint_probability(probs, sub_corr)
                ev = p * odds - 1
                if ev >= ev_threshold:
                    combo = candidate
                    added = True
                    break
        probs = [matches[i].prob for i in combo]
        odds = float(np.prod([matches[i].odds for i in combo]))
        sub_corr = corr_matrix[np.ix_(combo, combo)] if corr_matrix is not None else None
        p = joint_probability(probs, sub_corr)
        ev = p * odds - 1
        parlays.append((tuple(combo), ev))

    # remove duplicates while keeping best EV
    unique = {}
    for combo, ev in parlays:
        if combo not in unique or ev > unique[combo]:
            unique[combo] = ev
    return list(unique.items())

