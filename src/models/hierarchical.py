from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.config import HALF_LIFE_DAYS
from src.models.params import posterior_arrays


@dataclass
class Design:
    home_idx: np.ndarray
    away_idx: np.ndarray
    league_idx: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    weights: np.ndarray
    teams: list[str]
    leagues: list[str]
    membership: np.ndarray


def build_design(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
                 half_life: float = HALF_LIFE_DAYS) -> Design:
    df = matches.dropna(subset=["home", "away", "hg", "ag"]).copy()
    as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max()
    df = df[df["date"] <= as_of]
    teams = sorted(set(df["home"]) | set(df["away"]))
    leagues = sorted(df["league"].unique())
    t_index = {t: i for i, t in enumerate(teams)}
    l_index = {l: i for i, l in enumerate(leagues)}

    home_idx = df["home"].map(t_index).to_numpy()
    away_idx = df["away"].map(t_index).to_numpy()
    league_idx = df["league"].map(l_index).to_numpy()

    membership = np.zeros((len(teams), len(leagues)))
    primary = df.groupby("home")["league"].agg(lambda s: s.value_counts().index[0])
    for team, league in primary.items():
        membership[t_index[team], l_index[league]] = 1.0
    orphan = membership.sum(axis=1) == 0
    membership[orphan, 0] = 1.0

    age = (as_of - df["date"]).dt.days.to_numpy().astype(float)
    weights = 0.5 ** (age / half_life)
    return Design(home_idx, away_idx, league_idx, df["hg"].to_numpy().astype(int),
                  df["ag"].to_numpy().astype(int), weights, teams, leagues, membership)


def _dixon_coles(hg, ag, lam, mu, rho):
    adj = pt.ones_like(lam)
    adj = pt.switch(pt.eq(hg, 0) & pt.eq(ag, 0), 1.0 - lam * mu * rho, adj)
    adj = pt.switch(pt.eq(hg, 0) & pt.eq(ag, 1), 1.0 + lam * rho, adj)
    adj = pt.switch(pt.eq(hg, 1) & pt.eq(ag, 0), 1.0 + mu * rho, adj)
    adj = pt.switch(pt.eq(hg, 1) & pt.eq(ag, 1), 1.0 - rho, adj)
    return pt.log(pt.clip(adj, 1e-6, np.inf))


def build_model(design: Design, fixed_sigma: float | None = None) -> pm.Model:
    n_teams, n_leagues = design.membership.shape
    counts = design.membership.sum(axis=0)
    with pm.Model() as model:
        if fixed_sigma is None:
            sigma_att = pm.HalfNormal("sigma_att", 0.5)
            sigma_def = pm.HalfNormal("sigma_def", 0.5)
        else:
            sigma_att = sigma_def = fixed_sigma
        att_raw = pm.Normal("att_raw", 0.0, sigma_att, shape=n_teams)
        def_raw = pm.Normal("def_raw", 0.0, sigma_def, shape=n_teams)

        member = pt.as_tensor_variable(design.membership)
        att = pm.Deterministic("att", att_raw - member @ (member.T @ att_raw / counts))
        dfn = pm.Deterministic("def", def_raw - member @ (member.T @ def_raw / counts))

        intercept = pm.Normal("intercept", 0.1, 0.5, shape=n_leagues)
        home_adv = pm.Normal("home_adv", 0.25, 0.25, shape=n_leagues)
        rho = pm.Normal("rho", 0.0, 0.1)

        log_lam = intercept[design.league_idx] + home_adv[design.league_idx] \
            + att[design.home_idx] - dfn[design.away_idx]
        log_mu = intercept[design.league_idx] + att[design.away_idx] - dfn[design.home_idx]
        lam = pt.exp(pt.clip(log_lam, -4, 3))
        mu = pt.exp(pt.clip(log_mu, -4, 3))

        logp = pm.logp(pm.Poisson.dist(lam), design.hg) \
            + pm.logp(pm.Poisson.dist(mu), design.ag) \
            + _dixon_coles(design.hg, design.ag, lam, mu, rho)
        pm.Potential("weighted_likelihood", (design.weights * logp).sum())
    return model


def fit(design: Design, draws: int = 1000, tune: int = 1000, chains: int = 4,
        target_accept: float = 0.9, seed: int = 42):
    with build_model(design):
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
                         target_accept=target_accept, random_seed=seed,
                         progressbar=True, compute_convergence_checks=True)


def ratings(result, design: Design) -> pd.DataFrame:
    flat = posterior_arrays(result)
    att, dfn = flat["att"], flat["def"]
    league_of = [design.leagues[int(np.argmax(row))] for row in design.membership]
    return pd.DataFrame({
        "team": design.teams,
        "league": league_of,
        "attack": att.mean(axis=1),
        "attack_sd": att.std(axis=1),
        "defence": dfn.mean(axis=1),
        "defence_sd": dfn.std(axis=1),
        "strength": att.mean(axis=1) + dfn.mean(axis=1),
    }).sort_values("strength", ascending=False).reset_index(drop=True)


def fit_map(design: Design, fixed_sigma: float = 0.45, seed: int = 42) -> dict[str, np.ndarray]:
    with build_model(design, fixed_sigma=fixed_sigma):
        point = pm.find_MAP(progressbar=False, seed=seed)
    out = {k: np.atleast_1d(np.asarray(v)) for k, v in point.items()}
    member, counts = design.membership, design.membership.sum(axis=0)
    for name, raw in (("att", "att_raw"), ("def", "def_raw")):
        if name not in out and raw in out:
            out[name] = out[raw] - member @ (member.T @ out[raw] / counts)
    return out
