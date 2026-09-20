import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import outcome_probabilities, score
from src.features.availability import normalise_player, resolve_players
from src.models.design import _blend
from src.models.dynamic import _ar1_operator


def test_blend_falls_back_to_goals_without_xg():
    goals = pd.Series([2.0, 1.0, 3.0])
    xg = pd.Series([1.4, np.nan, 2.1])
    out = _blend(goals, xg, 0.5)
    assert out[0] == pytest.approx(1.7)
    assert out[1] == 1.0
    assert out[2] == pytest.approx(2.55)


def test_blend_weight_zero_is_goals():
    goals = pd.Series([2.0, 0.0])
    assert list(_blend(goals, pd.Series([9.0, 9.0]), 0.0)) == [2.0, 0.0]


def test_ar1_operator_decays_with_lag():
    op = _ar1_operator(4, 0.9)
    assert op[0, 0] == 1.0
    assert op[3, 0] == pytest.approx(0.9 ** 3)
    assert op[0, 3] == 0.0
    assert np.allclose(np.triu(op, 1), 0)


def test_ar1_persistence_one_is_random_walk():
    op = _ar1_operator(5, 1.0)
    assert np.allclose(np.tril(op), np.tril(np.ones((5, 5))))


def test_outcome_probabilities_sum_to_one():
    lam = np.array([1.5, 0.8, 2.4])
    mu = np.array([1.1, 1.9, 0.6])
    probs = outcome_probabilities(lam, mu, -0.05)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[2, 0] > probs[2, 2]


def test_score_rewards_correct_confidence():
    actual = np.array([0, 1, 2])
    confident = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    hedged = np.full((3, 3), 1 / 3)
    assert score(confident, actual)["log_loss"] < score(hedged, actual)["log_loss"]
    assert score(confident, actual)["rps"] < score(hedged, actual)["rps"]


@pytest.mark.parametrize("reported,known", [
    ("E. Haaland", "Erling Haaland"), ("Martin Odegaard", "Martin Ødegaard"),
    ("Vinícius Júnior", "Vinicius Junior"), ("R. Lewandowski", "Robert Lewandowski"),
])
def test_injury_names_resolve(reported, known):
    assert resolve_players(pd.Series([reported]), pd.Series([known])) == {reported: known}


def test_unknown_player_is_not_forced_to_match():
    out = resolve_players(pd.Series(["Nobody Here"]), pd.Series(["Erling Haaland"]))
    assert out == {}


def test_normalise_player_strips_punctuation():
    assert normalise_player("N'Golo  Kanté") == "n golo kante"
