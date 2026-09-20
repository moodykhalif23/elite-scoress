import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import season_codes, season_label
from src.ingest.teams import canonical
from src.models import simulate


@pytest.mark.parametrize("a,b", [
    ("Man United", "Manchester United"), ("M'gladbach", "Borussia M.Gladbach"),
    ("Ath Madrid", "Atletico Madrid"), ("FC Koln", "FC Cologne"),
    ("Nott'm Forest", "Nottingham Forest"), ("Milan", "AC Milan"),
    ("RB Leipzig", "RasenBallsport Leipzig"), ("Hamburg", "Hamburger SV"),
])
def test_team_aliases_resolve(a, b):
    assert canonical(a) == canonical(b)


def test_distinct_teams_stay_distinct():
    assert canonical("Man United") != canonical("Man City")
    assert canonical("Real Madrid") != canonical("Ath Madrid")


def test_season_codes_span():
    codes = season_codes(2000, 2024)
    assert codes[0] == "0001" and codes[-1] == "2425"
    assert season_label("0001") == "2000/01"


def test_markets_are_a_distribution():
    m = simulate.markets(1.5, 1.1, rho=-0.05)
    assert abs(m["home"] + m["draw"] + m["away"] - 1) < 1e-6
    assert 0 <= m["btts"] <= 1
    assert m["over_1.5"] > m["over_2.5"] > m["over_3.5"]


def test_stronger_home_side_wins_more():
    weak = simulate.markets(1.0, 1.0)
    strong = simulate.markets(2.2, 0.8)
    assert strong["home"] > weak["home"]
    assert strong["exp_hg"] > weak["exp_hg"]


def test_expected_goals_track_rates():
    m = simulate.markets(1.7, 0.9)
    assert abs(m["exp_hg"] - 1.7) < 0.05
    assert abs(m["exp_ag"] - 0.9) < 0.05


def test_dixon_coles_shifts_low_scores():
    base = simulate._dc_matrix(1.3, 1.1, 0.0, 10)
    adj = simulate._dc_matrix(1.3, 1.1, -0.08, 10)
    assert adj[0, 0] != base[0, 0]
    assert abs(adj.sum() - 1) < 1e-9
