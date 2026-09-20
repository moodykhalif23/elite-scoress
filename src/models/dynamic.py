from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.design import Design
from src.models.hierarchical import _dixon_coles, ppml
from src.models.params import posterior_arrays

INNOVATION = 0.22
PERSISTENCE = 0.92
INITIAL = 0.40


def _ar1_operator(n_periods: int, phi: float) -> np.ndarray:
    lag = np.arange(n_periods)[:, None] - np.arange(n_periods)[None, :]
    return np.where(lag >= 0, phi ** np.clip(lag, 0, None), 0.0)


def _centre(values, membership, counts):
    member = pt.as_tensor_variable(membership)
    return values - (member @ (member.T @ values / counts[:, None]))


def build_model(design: Design, innovation: float = INNOVATION,
                persistence: float = PERSISTENCE, initial: float = INITIAL) -> pm.Model:
    counts = design.membership.sum(axis=0)
    shape = (design.n_periods, design.n_teams)
    with pm.Model() as model:
        scales = np.full(shape, innovation)
        scales[0, :] = initial
        att_eps = pm.Normal("att_eps", 0.0, scales, shape=shape)
        def_eps = pm.Normal("def_eps", 0.0, scales, shape=shape)

        operator = pt.as_tensor_variable(_ar1_operator(design.n_periods, persistence))
        att_path = operator @ att_eps
        def_path = operator @ def_eps
        att = pm.Deterministic("att_path", _centre(att_path.T, design.membership, counts).T)
        dfn = pm.Deterministic("def_path", _centre(def_path.T, design.membership, counts).T)

        intercept = pm.Normal("intercept", 0.1, 0.5, shape=design.n_leagues)
        home_adv = pm.Normal("home_adv", 0.25, 0.25, shape=design.n_leagues)
        rho = pm.Normal("rho", 0.0, 0.1)

        rows, home, away = design.period_idx, design.home_idx, design.away_idx
        base = intercept[design.league_idx]
        lam = pt.exp(pt.clip(base + home_adv[design.league_idx]
                             + att[rows, home] - dfn[rows, away], -4, 3))
        mu = pt.exp(pt.clip(base + att[rows, away] - dfn[rows, home], -4, 3))

        logp = ppml(design.target_h, lam) + ppml(design.target_a, mu) \
            + _dixon_coles(design.hg, design.ag, lam, mu, rho)
        pm.Potential("likelihood", (design.weights * logp).sum())
    return model


def fit_map(design: Design, innovation: float = INNOVATION, persistence: float = PERSISTENCE,
            initial: float = INITIAL, seed: int = 42) -> dict:
    with build_model(design, innovation, persistence, initial):
        point = pm.find_MAP(progressbar=False, seed=seed)
    out = {k: np.atleast_1d(np.asarray(v)) for k, v in point.items()}
    out["att"] = out["att_path"][-1]
    out["def"] = out["def_path"][-1]
    out["persistence"] = np.array([persistence])
    return out


def rates(params: dict, design: Design, home_idx: np.ndarray, away_idx: np.ndarray,
          league_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    phi = float(np.ravel(params.get("persistence", [PERSISTENCE]))[0])
    att, dfn = params["att"] * phi, params["def"] * phi
    base = params["intercept"][league_idx]
    lam = np.exp(base + params["home_adv"][league_idx] + att[home_idx] - dfn[away_idx])
    mu = np.exp(base + att[away_idx] - dfn[home_idx])
    return lam, mu, float(np.ravel(params["rho"])[0])


def trajectory(params: dict, design: Design, team: str) -> pd.DataFrame:
    idx = design.teams.index(team)
    return pd.DataFrame({
        "season": design.periods,
        "attack": params["att_path"][:, idx],
        "defence": params["def_path"][:, idx],
    })
