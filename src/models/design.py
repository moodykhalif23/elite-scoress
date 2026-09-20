from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import HALF_LIFE_DAYS, XG_WEIGHT


@dataclass
class Design:
    home_idx: np.ndarray
    away_idx: np.ndarray
    league_idx: np.ndarray
    period_idx: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    target_h: np.ndarray
    target_a: np.ndarray
    weights: np.ndarray
    teams: list[str]
    leagues: list[str]
    periods: list[str]
    membership: np.ndarray

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_leagues(self) -> int:
        return len(self.leagues)

    @property
    def n_periods(self) -> int:
        return len(self.periods)


def _blend(goals: pd.Series, xg: pd.Series, weight: float) -> np.ndarray:
    out = goals.to_numpy().astype(float)
    if weight <= 0:
        return out
    has_xg = xg.notna().to_numpy()
    blended = weight * xg.to_numpy() + (1 - weight) * out
    return np.where(has_xg, blended, out)


def build_design(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
                 half_life: float = HALF_LIFE_DAYS, xg_weight: float = XG_WEIGHT,
                 decay: bool = True) -> Design:
    df = matches.dropna(subset=["home", "away", "hg", "ag"]).copy()
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    df = df[df["date"] <= as_of].sort_values("date")

    teams = sorted(set(df["home"]) | set(df["away"]))
    leagues = sorted(df["league"].unique())
    periods = sorted(df["season"].unique())
    t_index = {t: i for i, t in enumerate(teams)}
    l_index = {l: i for i, l in enumerate(leagues)}
    p_index = {p: i for i, p in enumerate(periods)}

    membership = np.zeros((len(teams), len(leagues)))
    primary = df.groupby("home")["league"].agg(lambda s: s.value_counts().index[0])
    for team, league in primary.items():
        membership[t_index[team], l_index[league]] = 1.0
    membership[membership.sum(axis=1) == 0, 0] = 1.0

    age = (as_of - df["date"]).dt.days.to_numpy().astype(float)
    weights = 0.5 ** (age / half_life) if decay else np.ones(len(df))

    return Design(
        home_idx=df["home"].map(t_index).to_numpy(),
        away_idx=df["away"].map(t_index).to_numpy(),
        league_idx=df["league"].map(l_index).to_numpy(),
        period_idx=df["season"].map(p_index).to_numpy(),
        hg=df["hg"].to_numpy().astype(int),
        ag=df["ag"].to_numpy().astype(int),
        target_h=_blend(df["hg"], df["xg_h"], xg_weight),
        target_a=_blend(df["ag"], df["xg_a"], xg_weight),
        weights=weights,
        teams=teams, leagues=leagues, periods=periods, membership=membership,
    )
