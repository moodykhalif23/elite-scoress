from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED
from src.ingest import football_data, understat
from src.ingest.teams import canonical

FORM_WINDOW = 6
POINTS = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}


def _long_form(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches.assign(team=matches["home"], opponent=matches["away"], venue="H",
                          gf=matches["hg"], ga=matches["ag"],
                          xgf=matches.get("xg_h"), xga=matches.get("xg_a"))
    away = matches.assign(team=matches["away"], opponent=matches["home"], venue="A",
                          gf=matches["ag"], ga=matches["hg"],
                          xgf=matches.get("xg_a"), xga=matches.get("xg_h"))
    cols = ["match_id", "date", "league", "season", "team", "opponent", "venue", "gf", "ga",
            "xgf", "xga", "result"]
    long = pd.concat([home[cols], away[cols]], ignore_index=True)
    long["points"] = [POINTS[r][0] if v == "H" else POINTS[r][1]
                      for r, v in zip(long["result"], long["venue"])]
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def _rolling(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby("team", sort=False)
    out = long[["match_id", "team", "venue"]].copy()
    for col, name in [("points", "form_pts"), ("gf", "form_gf"), ("ga", "form_ga"),
                      ("xgf", "form_xgf"), ("xga", "form_xga")]:
        shifted = grouped[col].shift(1)
        out[name] = shifted.groupby(long["team"]).rolling(FORM_WINDOW, min_periods=1).mean() \
            .reset_index(level=0, drop=True)
    out["rest_days"] = grouped["date"].diff().dt.days
    out["played"] = grouped.cumcount()
    return out


def _h2h(long: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = long[["match_id", "team", "opponent", "venue", "points"]].copy()
    frame["pair"] = [f"{t}|{o}" for t, o in zip(frame["team"], frame["opponent"])]
    frame = frame.sort_values(["pair", "match_id"])
    grouped = frame.groupby("pair", sort=False)
    frame["h2h_played"] = grouped.cumcount()
    frame["h2h_pts"] = grouped["points"].shift(1).groupby(frame["pair"]) \
        .rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    home = frame[frame["venue"] == "H"]
    return home[["match_id", "h2h_played", "h2h_pts"]].rename(
        columns={"h2h_pts": "h2h_home_pts"})


def build(refresh: bool = False, with_xg: bool = True) -> pd.DataFrame:
    matches = football_data.load_all(refresh=refresh)
    matches["home"] = matches["home"].map(canonical)
    matches["away"] = matches["away"].map(canonical)
    matches = matches[matches["home"] != matches["away"]].copy()
    matches = matches.sort_values("date").reset_index(drop=True)
    matches["match_id"] = np.arange(len(matches))

    if with_xg:
        xg = understat.load_all("matches", refresh=refresh)
        if not xg.empty:
            xg["home"] = xg["home"].map(canonical)
            xg["away"] = xg["away"].map(canonical)
            xg["date"] = pd.to_datetime(xg["date"]).dt.normalize()
            matches["_d"] = matches["date"].dt.normalize()
            xg = xg[["date", "home", "away", "xg_h", "xg_a"]].rename(columns={"date": "_d"})
            matches = matches.merge(xg.drop_duplicates(["_d", "home", "away"]),
                                    on=["_d", "home", "away"], how="left").drop(columns="_d")
    for col in ("xg_h", "xg_a"):
        if col not in matches.columns:
            matches[col] = np.nan

    long = _long_form(matches)
    roll = _rolling(long)
    home_roll = roll[roll["venue"] == "H"].drop(columns=["team", "venue"]) \
        .add_prefix("h_").rename(columns={"h_match_id": "match_id"})
    away_roll = roll[roll["venue"] == "A"].drop(columns=["team", "venue"]) \
        .add_prefix("a_").rename(columns={"a_match_id": "match_id"})

    out = matches.merge(home_roll, on="match_id", how="left") \
                 .merge(away_roll, on="match_id", how="left") \
                 .merge(_h2h(long), on="match_id", how="left")
    out["days_ago"] = (out["date"].max() - out["date"]).dt.days
    return out


def save(df: pd.DataFrame, name: str = "matches.parquet") -> str:
    path = PROCESSED / name
    df.to_parquet(path, index=False)
    return str(path)


def load(name: str = "matches.parquet") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / name)


def coverage(matches: pd.DataFrame) -> pd.DataFrame:
    grouped = matches.groupby("league")
    return pd.DataFrame({
        "matches": grouped.size(),
        "seasons": grouped["season"].nunique(),
        "teams": grouped.apply(lambda g: len(set(g["home"]) | set(g["away"])), include_groups=False),
        "first": grouped["date"].min().dt.date,
        "last": grouped["date"].max().dt.date,
        "xg_pct": (grouped["xg_h"].apply(lambda s: s.notna().mean()) * 100).round(1),
        "odds_pct": (grouped["odds_h"].apply(lambda s: s.notna().mean()) * 100).round(1),
    }).reset_index()
