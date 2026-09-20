from __future__ import annotations

import io

import pandas as pd
import requests

from src.config import LEAGUES, RAW
from src.ingest.football_data import _best_odds, _read_csv
from src.ingest.teams import canonical

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def upcoming(refresh: bool = True) -> pd.DataFrame:
    path = RAW / "fixtures.csv"
    if refresh or not path.exists():
        try:
            resp = requests.get(FIXTURES_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except requests.RequestException:
            if not path.exists():
                return pd.DataFrame()
    df = _read_csv(path)
    df = df[df["Div"].isin(LEAGUES)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "league": df["Div"],
        "home": df["HomeTeam"].map(canonical),
        "away": df["AwayTeam"].map(canonical),
        "home_name": df["HomeTeam"],
        "away_name": df["AwayTeam"],
        "date": pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"),
        "time": df.get("Time"),
    }).join(_best_odds(df))
    return out.dropna(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
