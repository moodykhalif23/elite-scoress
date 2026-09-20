from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import MAX_GOALS
from src.models.design import build_design

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_probabilities(lam: np.ndarray, mu: np.ndarray, rho: float,
                          max_goals: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(max_goals + 1)
    grid = poisson.pmf(k[None, :], lam[:, None])[:, :, None] \
        * poisson.pmf(k[None, :], mu[:, None])[:, None, :]
    grid[:, 0, 0] *= 1 - lam * mu * rho
    grid[:, 0, 1] *= 1 + lam * rho
    grid[:, 1, 0] *= 1 + mu * rho
    grid[:, 1, 1] *= 1 - rho
    grid = np.clip(grid, 0, None)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    diff = k[:, None] - k[None, :]
    return np.stack([grid[:, diff > 0].sum(1), grid[:, diff == 0].sum(1),
                     grid[:, diff < 0].sum(1)], axis=1)


def score(probs: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    n = len(actual)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), actual] = 1.0
    return {
        "log_loss": float(-np.log(np.clip(probs[np.arange(n), actual], 1e-12, 1)).mean()),
        "rps": float(((probs.cumsum(1) - onehot.cumsum(1))[:, :2] ** 2).sum(1).mean() / 2),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "accuracy": float((probs.argmax(1) == actual).mean()),
        "n": n,
    }


def fold(matches: pd.DataFrame, season: str, model, fit_kwargs: dict,
         design_kwargs: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    train = matches[matches["season"] < season]
    test = matches[matches["season"] == season]
    if train.empty or test.empty:
        return None
    design = build_design(train, as_of=train["date"].max(), **design_kwargs)
    params = model.fit_map(design, **fit_kwargs)

    t_index = {t: i for i, t in enumerate(design.teams)}
    l_index = {l: i for i, l in enumerate(design.leagues)}
    known = test["home"].isin(t_index) & test["away"].isin(t_index) \
        & test["league"].isin(l_index)
    test = test[known]
    if test.empty:
        return None
    lam, mu, rho = model.rates(params, design, test["home"].map(t_index).to_numpy(),
                               test["away"].map(t_index).to_numpy(),
                               test["league"].map(l_index).to_numpy())
    probs = outcome_probabilities(lam, mu, rho)
    return probs, test["result"].map(OUTCOME_INDEX).to_numpy(), test


def walk_forward(matches: pd.DataFrame, seasons: list[str], model,
                 fit_kwargs: dict | None = None,
                 design_kwargs: dict | None = None) -> tuple[dict, pd.DataFrame]:
    probs, actual, frames = [], [], []
    for season in seasons:
        result = fold(matches, season, model, fit_kwargs or {}, design_kwargs or {})
        if result is None:
            continue
        probs.append(result[0])
        actual.append(result[1])
        frames.append(result[2])
    if not probs:
        return {}, pd.DataFrame()
    stacked = np.vstack(probs)
    actual = np.concatenate(actual)
    out = pd.concat(frames, ignore_index=True)
    out[["p_home", "p_draw", "p_away"]] = stacked
    return score(stacked, actual), out
