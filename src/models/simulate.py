from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.config import MAX_GOALS
from src.models.params import posterior_arrays


def _dc_matrix(lam: float, mu: float, rho: float, max_goals: int) -> np.ndarray:
    home = poisson.pmf(np.arange(max_goals + 1), lam)
    away = poisson.pmf(np.arange(max_goals + 1), mu)
    grid = np.outer(home, away)
    grid[0, 0] *= 1.0 - lam * mu * rho
    grid[0, 1] *= 1.0 + lam * rho
    grid[1, 0] *= 1.0 + mu * rho
    grid[1, 1] *= 1.0 - rho
    grid = np.clip(grid, 0.0, None)
    total = grid.sum()
    return grid / total if total > 0 else grid


def markets(lam: float, mu: float, rho: float = 0.0,
            max_goals: int = MAX_GOALS) -> dict[str, float]:
    grid = _dc_matrix(lam, mu, rho, max_goals)
    idx = np.arange(max_goals + 1)
    diff = idx[:, None] - idx[None, :]
    total = idx[:, None] + idx[None, :]
    both = (idx[:, None] > 0) & (idx[None, :] > 0)
    out = {
        "home": float(grid[diff > 0].sum()),
        "draw": float(grid[diff == 0].sum()),
        "away": float(grid[diff < 0].sum()),
        "btts": float(grid[both].sum()),
        "exp_hg": float((grid.sum(axis=1) * idx).sum()),
        "exp_ag": float((grid.sum(axis=0) * idx).sum()),
    }
    for line in (1.5, 2.5, 3.5):
        out[f"over_{line}"] = float(grid[total > line].sum())
    top = np.unravel_index(np.argmax(grid), grid.shape)
    out["top_score"] = f"{top[0]}-{top[1]}"
    out["top_score_p"] = float(grid[top])
    return out


def _rates(idata, design, home: str, away: str, league: str,
           adjust: tuple[float, float] = (0.0, 0.0)):
    t_index = {t: i for i, t in enumerate(design.teams)}
    l_index = {l: i for i, l in enumerate(design.leagues)}
    if home not in t_index or away not in t_index or league not in l_index:
        return None
    h, a, l = t_index[home], t_index[away], l_index[league]
    flat = posterior_arrays(idata)
    base = flat["intercept"][l]
    lam = np.exp(base + flat["home_adv"][l] + flat["att"][h] - flat["def"][a] + adjust[0])
    mu = np.exp(base + flat["att"][a] - flat["def"][h] + adjust[1])
    return lam, mu, np.broadcast_to(flat["rho"].ravel(), lam.shape)


def predict_match(idata, design, home: str, away: str, league: str,
                  adjust: tuple[float, float] = (0.0, 0.0),
                  max_goals: int = MAX_GOALS, sample: int = 400) -> dict | None:
    rates = _rates(idata, design, home, away, league, adjust)
    if rates is None:
        return None
    lam, mu, rho = rates
    take = np.linspace(0, len(lam) - 1, min(sample, len(lam))).astype(int)
    rows = [markets(float(lam[i]), float(mu[i]), float(rho[i]), max_goals) for i in take]
    frame = pd.DataFrame(rows)
    numeric = frame.drop(columns=["top_score"])
    out = numeric.mean().to_dict()
    out["home_lo"], out["home_hi"] = numeric["home"].quantile([0.05, 0.95])
    out["draw_lo"], out["draw_hi"] = numeric["draw"].quantile([0.05, 0.95])
    out["away_lo"], out["away_hi"] = numeric["away"].quantile([0.05, 0.95])
    out["top_score"] = frame["top_score"].mode().iloc[0]
    out["scoreline_grid"] = _dc_matrix(float(lam.mean()), float(mu.mean()),
                                       float(rho.mean()), max_goals)
    return out


def predict_fixtures(idata, design, fixtures: pd.DataFrame,
                     adjustments: dict[int, tuple[float, float]] | None = None) -> pd.DataFrame:
    adjustments = adjustments or {}
    rows = []
    for i, fx in fixtures.reset_index(drop=True).iterrows():
        res = predict_match(idata, design, fx["home"], fx["away"], fx["league"],
                            adjustments.get(i, (0.0, 0.0)))
        if res is None:
            continue
        res.pop("scoreline_grid", None)
        rows.append({**fx.to_dict(), **res})
    return pd.DataFrame(rows)
