from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import pandas as pd
import requests

from src.config import LEAGUES, PROCESSED, RAW, UNDERSTAT_URL, current_season_start

FIRST_YEAR = 2014
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def fetch_season(league: str, year: int, refresh: bool = False) -> dict | None:
    path = RAW / "understat" / f"{league.replace(' ', '_')}_{year}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    url = UNDERSTAT_URL.format(league=quote(league), year=year)
    headers = {**HEADERS, "Referer": f"https://understat.com/league/{quote(league)}/{year}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "dates" not in data:
        return None
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def matches(league: str, year: int, refresh: bool = False) -> pd.DataFrame:
    data = fetch_season(league, year, refresh)
    if not data:
        return pd.DataFrame()
    rows = [{
        "date": m["datetime"],
        "home": m["h"]["title"],
        "away": m["a"]["title"],
        "xg_h": float(m["xG"]["h"]),
        "xg_a": float(m["xG"]["a"]),
    } for m in data["dates"] if m.get("isResult")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def players(league: str, year: int, refresh: bool = False) -> pd.DataFrame:
    data = fetch_season(league, year, refresh)
    if not data or not data.get("players"):
        return pd.DataFrame()
    df = pd.DataFrame(data["players"])
    rename = {"player_name": "player", "team_title": "team", "time": "minutes",
              "xG": "xg", "xA": "xa", "npxG": "npxg", "xGChain": "xg_chain",
              "xGBuildup": "xg_buildup"}
    df = df.rename(columns=rename)
    numeric = ["minutes", "games", "goals", "assists", "xg", "xa", "npxg", "shots",
               "key_passes", "xg_chain", "xg_buildup"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = ["player", "team", "position"] + [c for c in numeric if c in df.columns]
    out = df[keep].copy()
    out["season_start"] = year
    return out


def _history_path(kind: str):
    return PROCESSED / f"history_understat_{kind}.parquet"


def _raw_path(league: str, year: int):
    return RAW / "understat" / f"{league.replace(' ', '_')}_{year}.json"


def _history_is_stale(kind: str, live: int) -> bool:
    path = _history_path(kind)
    if not path.exists():
        return True
    stamp = path.stat().st_mtime
    return any(_raw_path(lg.understat, y).stat().st_mtime > stamp
               for lg in LEAGUES.values() for y in range(FIRST_YEAR, live)
               if _raw_path(lg.understat, y).exists())


def load_all(kind: str = "matches", refresh: bool = False, pause: float = 0.4,
             current_only: bool = False) -> pd.DataFrame:
    getter = matches if kind == "matches" else players
    live = current_season_start()
    frames = []

    path = _history_path(kind)
    if not refresh and not _history_is_stale(kind, live):
        frames.append(pd.read_parquet(path))
    else:
        past = []
        for league in LEAGUES.values():
            for year in range(FIRST_YEAR, live):
                df = getter(league.understat, year, refresh)
                if df.empty:
                    continue
                df["league"] = league.code
                df["season_start"] = year
                past.append(df)
                if refresh:
                    time.sleep(pause)
        history = pd.concat(past, ignore_index=True) if past else pd.DataFrame()
        if not history.empty:
            history.to_parquet(path, index=False)
        frames.append(history)

    leagues = list(LEAGUES.values())
    with ThreadPoolExecutor(max_workers=len(leagues)) as pool:
        current = pool.map(lambda lg: (lg, getter(lg.understat, live,
                                                  refresh or current_only)), leagues)
    for league, df in current:
        if df.empty:
            continue
        df["league"] = league.code
        df["season_start"] = live
        frames.append(df)

    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
