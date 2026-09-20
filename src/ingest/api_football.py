from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

from src.config import RAW

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"E0": 39, "SP1": 140, "I1": 135, "D1": 78, "UCL": 2, "UEL": 3}
CACHE = RAW / "api-football"
THROTTLE = 0.25


class Budget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def api_key() -> str | None:
    return os.environ.get("API_FOOTBALL_KEY")


def available() -> bool:
    return bool(api_key())


def _cache_path(endpoint: str, params: dict):
    stem = endpoint.strip("/").replace("/", "_")
    tag = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    return CACHE / stem / f"{tag}.json"


def fetch(endpoint: str, params: dict, budget: Budget | None = None) -> dict | None:
    path = _cache_path(endpoint, params)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    key = api_key()
    if not key or (budget is not None and not budget.spend()):
        return None
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", params=params, timeout=30,
                            headers={"x-apisports-key": key})
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return None
    if payload.get("errors"):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(THROTTLE)
    return payload


def fixtures(league: str, season: int, budget: Budget | None = None) -> pd.DataFrame:
    payload = fetch("/fixtures", {"league": LEAGUE_IDS[league], "season": season}, budget)
    if not payload:
        return pd.DataFrame()
    rows = [{
        "fixture_id": item["fixture"]["id"],
        "date": item["fixture"]["date"],
        "home": item["teams"]["home"]["name"],
        "away": item["teams"]["away"]["name"],
        "league": league,
        "season_start": season,
    } for item in payload.get("response", [])]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True).dt.tz_localize(None)
    return df


def lineups(fixture_id: int, budget: Budget | None = None) -> pd.DataFrame:
    payload = fetch("/fixtures/lineups", {"fixture": fixture_id}, budget)
    if not payload:
        return pd.DataFrame()
    rows = []
    for side in payload.get("response", []):
        team = side["team"]["name"]
        for slot, group in (("start", side.get("startXI") or []),
                            ("bench", side.get("substitutes") or [])):
            for entry in group:
                player = entry.get("player", {})
                rows.append({"fixture_id": fixture_id, "team": team,
                             "player": player.get("name"), "role": slot,
                             "position": player.get("pos"),
                             "formation": side.get("formation")})
    return pd.DataFrame(rows)


def injuries(league: str, season: int, budget: Budget | None = None) -> pd.DataFrame:
    payload = fetch("/injuries", {"league": LEAGUE_IDS[league], "season": season}, budget)
    if not payload:
        return pd.DataFrame()
    rows = [{
        "fixture_id": item.get("fixture", {}).get("id"),
        "date": item.get("fixture", {}).get("date"),
        "team": item["team"]["name"],
        "player": item["player"]["name"],
        "reason": item["player"].get("reason"),
        "type": item["player"].get("type"),
        "league": league,
        "season_start": season,
    } for item in payload.get("response", [])]
    return pd.DataFrame(rows)
