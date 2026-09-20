from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.design import Design, build_design  # noqa: F401
from src.models.params import posterior_arrays

MAP_SIGMA = 0.45
PAIR_SIGMA = 0.0
DERBY = False


def _dixon_coles(hg, ag, lam, mu, rho):
    adj = pt.ones_like(lam)
    adj = pt.switch(pt.eq(hg, 0) & pt.eq(ag, 0), 1.0 - lam * mu * rho, adj)
    adj = pt.switch(pt.eq(hg, 0) & pt.eq(ag, 1), 1.0 + lam * rho, adj)
    adj = pt.switch(pt.eq(hg, 1) & pt.eq(ag, 0), 1.0 + mu * rho, adj)
    adj = pt.switch(pt.eq(hg, 1) & pt.eq(ag, 1), 1.0 - rho, adj)
    return pt.log(pt.clip(adj, 1e-6, np.inf))


def ppml(target, rate):
    return target * pt.log(rate) - rate


def _centre(values, membership, counts):
    member = pt.as_tensor_variable(membership)
    return values - member @ (member.T @ values / counts)


def build_model(design: Design, fixed_sigma: float | None = None,
                pair_sigma: float = PAIR_SIGMA, derby: bool = DERBY) -> pm.Model:
    counts = design.membership.sum(axis=0)
    with pm.Model() as model:
        if fixed_sigma is None:
            sigma_att = pm.HalfNormal("sigma_att", 0.5)
            sigma_def = pm.HalfNormal("sigma_def", 0.5)
        else:
            sigma_att = sigma_def = fixed_sigma

        att_raw = pm.Normal("att_raw", 0.0, sigma_att, shape=design.n_teams)
        def_raw = pm.Normal("def_raw", 0.0, sigma_def, shape=design.n_teams)
        att = pm.Deterministic("att", _centre(att_raw, design.membership, counts))
        dfn = pm.Deterministic("def", _centre(def_raw, design.membership, counts))

        intercept = pm.Normal("intercept", 0.1, 0.5, shape=design.n_leagues)
        home_adv = pm.Normal("home_adv", 0.25, 0.25, shape=design.n_leagues)
        rho = pm.Normal("rho", 0.0, 0.1)

        base = intercept[design.league_idx]
        log_lam = base + home_adv[design.league_idx] \
            + att[design.home_idx] - dfn[design.away_idx]
        log_mu = base + att[design.away_idx] - dfn[design.home_idx]

        if pair_sigma > 0:
            level = pm.Normal("pair_level", 0.0, pair_sigma, shape=design.n_pairs)
            tilt = pm.Normal("pair_tilt", 0.0, pair_sigma, shape=design.n_pairs)
            swing = design.pair_orient * tilt[design.pair_idx]
            log_lam = log_lam + level[design.pair_idx] + swing
            log_mu = log_mu + level[design.pair_idx] - swing

        if derby:
            derby_level = pm.Normal("derby_level", 0.0, 0.15)
            derby_home = pm.Normal("derby_home", 0.0, 0.15)
            log_lam = log_lam + design.is_derby * (derby_level + derby_home)
            log_mu = log_mu + design.is_derby * derby_level

        lam = pt.exp(pt.clip(log_lam, -4, 3))
        mu = pt.exp(pt.clip(log_mu, -4, 3))

        logp = ppml(design.target_h, lam) + ppml(design.target_a, mu) \
            + _dixon_coles(design.hg, design.ag, lam, mu, rho)
        pm.Potential("weighted_likelihood", (design.weights * logp).sum())
    return model


def fit(design: Design, draws: int = 1000, tune: int = 1000, chains: int = 4,
        target_accept: float = 0.9, seed: int = 42):
    with build_model(design):
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=min(chains, 4),
                         target_accept=target_accept, random_seed=seed,
                         progressbar=True, compute_convergence_checks=True)


def fit_map(design: Design, fixed_sigma: float = MAP_SIGMA, pair_sigma: float = PAIR_SIGMA,
            derby: bool = DERBY, seed: int = 42) -> dict:
    with build_model(design, fixed_sigma=fixed_sigma, pair_sigma=pair_sigma, derby=derby):
        point = pm.find_MAP(progressbar=False, seed=seed)
    out = {k: np.atleast_1d(np.asarray(v)) for k, v in point.items()}
    counts = design.membership.sum(axis=0)
    for name, raw in (("att", "att_raw"), ("def", "def_raw")):
        if name not in out and raw in out:
            out[name] = out[raw] - design.membership @ (design.membership.T @ out[raw] / counts)
    return out


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


def rates(params: dict, design: Design, idx: dict) -> tuple[np.ndarray, np.ndarray, float]:
    att, dfn = params["att"], params["def"]
    base = params["intercept"][idx["league_idx"]]
    log_lam = base + params["home_adv"][idx["league_idx"]] \
        + att[idx["home_idx"]] - dfn[idx["away_idx"]]
    log_mu = base + att[idx["away_idx"]] - dfn[idx["home_idx"]]

    if "pair_level" in params:
        seen = idx["pair_seen"]
        level = params["pair_level"][idx["pair_idx"]] * seen
        swing = params["pair_tilt"][idx["pair_idx"]] * seen * idx["pair_orient"]
        log_lam = log_lam + level + swing
        log_mu = log_mu + level - swing

    if "derby_level" in params:
        derby_level = float(np.ravel(params["derby_level"])[0])
        derby_home = float(np.ravel(params["derby_home"])[0])
        log_lam = log_lam + idx["is_derby"] * (derby_level + derby_home)
        log_mu = log_mu + idx["is_derby"] * derby_level

    return np.exp(log_lam), np.exp(log_mu), float(np.ravel(params["rho"])[0])
