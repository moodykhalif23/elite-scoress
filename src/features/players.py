from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED, current_season_start
from src.ingest import understat
from src.ingest.teams import canonical

REPLACEMENT_LEVEL = 0.65
DEFENSIVE = {"D", "GK"}


def _is_defensive(position: str) -> bool:
    return bool(set(str(position).split()) & DEFENSIVE)


def build_values(seasons_back: int = 2, refresh: bool = False,
                 current_only: bool = False) -> pd.DataFrame:
    players = understat.load_all("players", refresh=refresh, current_only=current_only)
    if players.empty:
        return pd.DataFrame()
    latest = current_season_start()
    players = players[players["season_start"] >= latest - seasons_back].copy()
    players["team"] = players["team"].map(canonical)
    players["contribution"] = players["xg"].fillna(0) + players["xa"].fillna(0)

    recency = 0.5 ** ((latest - players["season_start"]) / 1.0)
    players["w_contribution"] = players["contribution"] * recency
    players["w_minutes"] = players["minutes"].fillna(0) * recency

    agg = players.groupby(["league", "team", "player"], as_index=False).agg(
        minutes=("w_minutes", "sum"), contribution=("w_contribution", "sum"),
        position=("position", "first"))
    agg["defensive"] = agg["position"].map(_is_defensive)

    team_att = agg.groupby(["league", "team"])["contribution"].transform("sum")
    def_minutes = agg["minutes"].where(agg["defensive"], 0.0)
    agg["_def_minutes"] = def_minutes
    team_def = agg.groupby(["league", "team"])["_def_minutes"].transform("sum")

    agg["att_share"] = np.where(team_att > 0, agg["contribution"] / team_att, 0.0)
    agg["def_share"] = np.where(team_def > 0, agg["_def_minutes"] / team_def, 0.0)
    agg["per90"] = np.where(agg["minutes"] > 0, agg["contribution"] / (agg["minutes"] / 90), 0.0)
    return agg.drop(columns="_def_minutes").sort_values(
        ["team", "att_share"], ascending=[True, False]).reset_index(drop=True)


def team_adjustment(values: pd.DataFrame, team: str, unavailable: list[str],
                    replacement: float = REPLACEMENT_LEVEL) -> tuple[float, float]:
    if values.empty or not unavailable:
        return 0.0, 0.0
    squad = values[values["team"] == team]
    missing = squad[squad["player"].isin(unavailable)]
    if missing.empty:
        return 0.0, 0.0
    gap = 1.0 - replacement
    att_loss = float(missing["att_share"].sum()) * gap
    def_loss = float(missing["def_share"].sum()) * gap
    att_delta = np.log(max(1.0 - att_loss, 0.35))
    def_delta = -np.log(max(1.0 - def_loss, 0.35))
    return float(att_delta), float(def_delta)


def save(values: pd.DataFrame, name: str = "player_values.parquet") -> str:
    path = PROCESSED / name
    values.to_parquet(path, index=False)
    return str(path)


def load(name: str = "player_values.parquet") -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame(columns=["league", "team", "player", "att_share", "def_share"])
    return pd.read_parquet(path)
