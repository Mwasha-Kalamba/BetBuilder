import sys
import time
import numpy as np
from pathlib import Path

# ensure root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bet_builder import Match, cartesian_parlays, greedy_parlays, joint_probability


def create_sample_matches(n):
    return [Match(f"M{i}", prob=0.55 + 0.01 * i, odds=1.9 + 0.05 * i) for i in range(n)]


def test_greedy_limits_combinations():
    matches = create_sample_matches(7)
    corr = np.zeros((7, 7))
    greedy = greedy_parlays(matches, ev_threshold=0.05, max_len=3, corr_matrix=corr)
    exhaustive = cartesian_parlays(matches, max_len=3, corr_matrix=corr)
    assert len(greedy) < len(exhaustive)


def test_runtime_improvement():
    matches = create_sample_matches(10)
    corr = np.zeros((10, 10))
    start = time.time()
    cartesian_parlays(matches, max_len=4, corr_matrix=corr)
    cart_time = time.time() - start

    start = time.time()
    greedy_parlays(matches, ev_threshold=0.05, max_len=4, corr_matrix=corr)
    greedy_time = time.time() - start

    assert greedy_time < cart_time


def test_correlation_adjustment():
    probs = [0.6, 0.6]
    corr_pos = np.array([[1.0, 0.2], [0.2, 1.0]])
    joint_pos = joint_probability(probs, corr_pos)
    independent = probs[0] * probs[1]
    assert joint_pos > independent

    corr_neg = np.array([[1.0, -0.2], [-0.2, 1.0]])
    joint_neg = joint_probability(probs, corr_neg)
    assert joint_neg < independent
