from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import HALF_LIFE_DAYS, XG_WEIGHT
from src.features.rivalry import derby_lookup


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
    pair_idx: np.ndarray
    pair_orient: np.ndarray
    is_derby: np.ndarray
    teams: list[str]
    leagues: list[str]
    periods: list[str]
    pairs: list[str]
    team_leagues: list[str]
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

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def index_fixtures(self, fixtures: pd.DataFrame) -> dict[str, np.ndarray]:
        teams = {t: i for i, t in enumerate(self.teams)}
        leagues = {l: i for i, l in enumerate(self.leagues)}
        pairs = {p: i for i, p in enumerate(self.pairs)}
        derbies = derby_lookup()
        keys = _pair_keys(fixtures["home"], fixtures["away"])
        known = [pairs.get(k, -1) for k in keys]
        return {
            "usable": (fixtures["home"].isin(teams) & fixtures["away"].isin(teams)
                       & fixtures["league"].isin(leagues)).to_numpy(),
            "home_idx": fixtures["home"].map(teams).fillna(0).to_numpy().astype(int),
            "away_idx": fixtures["away"].map(teams).fillna(0).to_numpy().astype(int),
            "league_idx": fixtures["league"].map(leagues).fillna(0).to_numpy().astype(int),
            "pair_idx": np.array([max(i, 0) for i in known]),
            "pair_seen": np.array([i >= 0 for i in known], dtype=float),
            "pair_orient": _orientation(fixtures["home"], fixtures["away"]),
            "is_derby": np.array([frozenset(pair) in derbies for pair in
                                  zip(fixtures["home"], fixtures["away"])], dtype=float),
        }


def _pair_keys(home: pd.Series, away: pd.Series) -> list[str]:
    return [" v ".join(sorted((h, a))) for h, a in zip(home, away)]


def _orientation(home: pd.Series, away: pd.Series) -> np.ndarray:
    return np.array([1.0 if h <= a else -1.0 for h, a in zip(home, away)])


def _blend(goals: pd.Series, xg: pd.Series, weight: float) -> np.ndarray:
    out = goals.to_numpy().astype(float)
    if weight <= 0:
        return out
    has_xg = xg.notna().to_numpy()
    blended = weight * xg.to_numpy() + (1 - weight) * out
    return np.where(has_xg, blended, out)


def _team_league_map(df: pd.DataFrame) -> dict[str, str]:
    if "home_league" not in df.columns:
        stacked = pd.concat([df[["home", "league"]].rename(columns={"home": "team"}),
                             df[["away", "league"]].rename(columns={"away": "team"})])
    else:
        stacked = pd.concat([df[["home", "home_league"]].rename(
            columns={"home": "team", "home_league": "league"}),
            df[["away", "away_league"]].rename(
                columns={"away": "team", "away_league": "league"})])
    stacked = stacked.dropna(subset=["league"])
    if stacked.empty:
        return {}
    return stacked.groupby("team")["league"].agg(lambda s: s.value_counts().index[0]).to_dict()


def build_design(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
                 half_life: float = HALF_LIFE_DAYS, xg_weight: float = XG_WEIGHT,
                 decay: bool = True, cross_league: bool | None = None) -> Design:
    df = matches.dropna(subset=["home", "away", "hg", "ag"]).copy()
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    df = df[df["date"] <= as_of].sort_values("date")

    teams = sorted(set(df["home"]) | set(df["away"]))
    leagues = sorted(df["league"].unique())
    periods = sorted(df["season"].unique())
    t_index = {t: i for i, t in enumerate(teams)}
    l_index = {l: i for i, l in enumerate(leagues)}
    p_index = {p: i for i, p in enumerate(periods)}

    team_league = _team_league_map(df)
    team_leagues = [team_league.get(t, "OTHER") for t in teams]
    if cross_league is None:
        cross_league = df["league"].isin(("UCL", "UEL")).any()

    if cross_league:
        membership = np.ones((len(teams), 1))
    else:
        groups = sorted(set(team_leagues))
        g_index = {g: i for i, g in enumerate(groups)}
        membership = np.zeros((len(teams), len(groups)))
        for i, league in enumerate(team_leagues):
            membership[i, g_index[league]] = 1.0

    pair_keys = _pair_keys(df["home"], df["away"])
    pairs = sorted(set(pair_keys))
    p_pair = {p: i for i, p in enumerate(pairs)}
    derbies = derby_lookup()

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
        pair_idx=np.array([p_pair[k] for k in pair_keys]),
        pair_orient=_orientation(df["home"], df["away"]),
        is_derby=np.array([frozenset(pair) in derbies
                           for pair in zip(df["home"], df["away"])], dtype=float),
        weights=weights,
        teams=teams, leagues=leagues, periods=periods, pairs=pairs,
        team_leagues=team_leagues, membership=membership,
    )
