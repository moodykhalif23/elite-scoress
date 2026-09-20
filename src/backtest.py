from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ARTIFACTS

OUTCOMES = ["p_home", "p_draw", "p_away"]


def implied_probabilities(odds: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / odds[["odds_h", "odds_d", "odds_a"]].to_numpy()
    return inv / inv.sum(axis=1, keepdims=True)


def walk_forward(matches: pd.DataFrame, start_season: str = "2015/16",
                 variant: str = "static") -> tuple[pd.DataFrame, dict]:
    from src import evaluate
    from src.cli import select

    model = select(variant)
    df = matches.dropna(subset=["hg", "ag"]).sort_values("date").reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    if start_season not in seasons:
        start_season = seasons[len(seasons) // 2]
    tested = seasons[seasons.index(start_season):]

    summary, out = evaluate.walk_forward(df, tested, model,
                                         design_kwargs={"decay": variant == "static"})
    if out.empty:
        return out, {}
    actual = out["result"].map(evaluate.OUTCOME_INDEX).to_numpy()
    model_probs = out[OUTCOMES].to_numpy()

    report = {"model": summary, "n": int(len(out))}
    has_odds = out[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1).to_numpy()
    if has_odds.sum() > 50:
        report["bookmaker"] = evaluate.score(implied_probabilities(out[has_odds]),
                                             actual[has_odds])
        report["model_on_odds_subset"] = evaluate.score(model_probs[has_odds],
                                                        actual[has_odds])
    return out, report


def edge_table(preds: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    df = preds.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    if df.empty:
        return df
    book = implied_probabilities(df)
    for i, name in enumerate(OUTCOMES):
        df[f"edge_{name[2:]}"] = df[name] - book[:, i]
    columns = [f"edge_{n[2:]}" for n in OUTCOMES]
    df["best_edge"] = df[columns].max(axis=1)
    df["best_pick"] = df[columns].idxmax(axis=1).str.replace("edge_", "")
    return df[df["best_edge"] >= threshold].sort_values("best_edge", ascending=False)


def save(preds: pd.DataFrame, name: str = "backtest.parquet") -> str:
    path = ARTIFACTS / name
    preds.to_parquet(path, index=False)
    return str(path)
