from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ARTIFACTS
from src.models import hierarchical, simulate

OUTCOMES = ["home", "draw", "away"]


def implied_probabilities(odds: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / odds[["odds_h", "odds_d", "odds_a"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)


def _scores(probs: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    log_loss = float(-np.log(np.clip(probs[np.arange(len(actual)), actual], eps, 1)).mean())
    brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
    rps = float(((probs.cumsum(axis=1) - onehot.cumsum(axis=1))[:, :2] ** 2).sum(axis=1).mean() / 2)
    hit = float((probs.argmax(axis=1) == actual).mean())
    return {"log_loss": log_loss, "brier": brier, "rps": rps, "accuracy": hit}


def walk_forward(matches: pd.DataFrame, start_season: str = "2015/16",
                 min_train_days: int = 730, method: str = "map") -> tuple[pd.DataFrame, dict]:
    df = matches.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    if start_season not in seasons:
        start_season = seasons[len(seasons) // 2]
    test_seasons = seasons[seasons.index(start_season):]

    records = []
    for season in test_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if train.empty or test.empty:
            continue
        if (train["date"].max() - train["date"].min()).days < min_train_days:
            continue
        design = hierarchical.build_design(train, as_of=train["date"].max())
        result = hierarchical.fit_map(design) if method == "map" else \
            hierarchical.fit(design, draws=500, tune=500, chains=2)
        preds = simulate.predict_fixtures(result, design, test)
        if preds.empty:
            continue
        preds["season_tested"] = season
        records.append(preds)

    if not records:
        return pd.DataFrame(), {}
    out = pd.concat(records, ignore_index=True)
    actual = out["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    model_probs = out[OUTCOMES].to_numpy()
    model_probs = model_probs / model_probs.sum(axis=1, keepdims=True)
    summary = {"model": _scores(model_probs, actual), "n": int(len(out))}

    has_odds = out[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
    if has_odds.sum() > 50:
        book = implied_probabilities(out[has_odds])
        summary["bookmaker"] = _scores(book, actual[has_odds])
        summary["model_on_odds_subset"] = _scores(model_probs[has_odds], actual[has_odds])
    return out, summary


def edge_table(preds: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    df = preds.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    if df.empty:
        return df
    book = implied_probabilities(df)
    for i, name in enumerate(OUTCOMES):
        df[f"edge_{name}"] = df[name] - book[:, i]
    df["best_edge"] = df[[f"edge_{n}" for n in OUTCOMES]].max(axis=1)
    df["best_pick"] = df[[f"edge_{n}" for n in OUTCOMES]].idxmax(axis=1).str.replace("edge_", "")
    return df[df["best_edge"] >= threshold].sort_values("best_edge", ascending=False)


def save(preds: pd.DataFrame, name: str = "backtest.parquet") -> str:
    path = ARTIFACTS / name
    preds.to_parquet(path, index=False)
    return str(path)
