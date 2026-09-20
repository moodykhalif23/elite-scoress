from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import PROCESSED, current_season_start
from src.ingest import api_football
from src.ingest.teams import canonical

TRANSLITERATE = str.maketrans({"ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l",
                               "æ": "ae", "Æ": "ae", "œ": "oe", "ð": "d", "þ": "th",
                               "ı": "i", "ß": "ss"})


def normalise_player(name: str) -> str:
    text = str(name).translate(TRANSLITERATE)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text)).strip()


def _surname_key(name: str) -> str:
    parts = normalise_player(name).split()
    if not parts:
        return ""
    return f"{parts[0][:1]}|{parts[-1]}" if len(parts) > 1 else parts[0]


def resolve_players(reported: pd.Series, known: pd.Series) -> dict[str, str]:
    exact = {normalise_player(n): n for n in known}
    surname = {}
    for n in known:
        surname.setdefault(_surname_key(n), n)
    resolved = {}
    for name in reported.dropna().unique():
        key = normalise_player(name)
        hit = exact.get(key) or surname.get(_surname_key(name))
        if hit:
            resolved[name] = hit
    return resolved


def current_injuries(budget_limit: int = 8, season: int | None = None) -> pd.DataFrame:
    if not api_football.available():
        return pd.DataFrame(columns=["league", "team", "player", "reason", "type"])
    season = season or current_season_start()
    budget = api_football.Budget(budget_limit)
    frames = [api_football.injuries(code, season, budget) for code in api_football.LEAGUE_IDS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["league", "team", "player", "reason", "type"])
    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canonical)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], format="ISO8601", utc=True,
                                     errors="coerce").dt.tz_localize(None)
        out = out.sort_values("date").drop_duplicates(["team", "player"], keep="last")
    return out.reset_index(drop=True)


def unavailable_for(injuries: pd.DataFrame, values: pd.DataFrame, team: str) -> list[str]:
    if injuries.empty or values.empty:
        return []
    reported = injuries[injuries["team"] == team]
    squad = values[values["team"] == team]
    if reported.empty or squad.empty:
        return []
    mapping = resolve_players(reported["player"], squad["player"])
    return sorted(set(mapping.values()))


def save(injuries: pd.DataFrame, name: str = "injuries.parquet") -> str:
    path = PROCESSED / name
    injuries.to_parquet(path, index=False)
    return str(path)


def load(name: str = "injuries.parquet") -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame(columns=["league", "team", "player", "reason", "type"])
    return pd.read_parquet(path)
