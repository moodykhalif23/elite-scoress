from __future__ import annotations

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.config import (FOOTBALL_DATA_URL, LEAGUES, PROCESSED, RAW,
                        current_season_code, season_codes, season_label)

CORE = {
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "hg",
    "FTAG": "ag",
    "FTR": "result",
    "HS": "hs",
    "AS": "as_",
    "HST": "hst",
    "AST": "ast",
    "HC": "hc",
    "AC": "ac",
    "HY": "hy",
    "AY": "ay",
    "HR": "hr",
    "AR": "ar",
}

ODDS_PREFERENCE = [("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"),
                   ("WHH", "WHD", "WHA"), ("BWH", "BWD", "BWA"), ("LBH", "LBD", "LBA"),
                   ("GBH", "GBD", "GBA"), ("IWH", "IWD", "IWA"), ("SBH", "SBD", "SBA")]


def _read_csv(source) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, low_memory=False,
                               on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "unreadable csv")


def _cache_path(code: str, season: str):
    return RAW / "football-data" / f"{code}_{season}.csv"


def _tidy_path(code: str, season: str):
    return PROCESSED / "seasons" / f"{code}_{season}.parquet"


def _is_fresh(tidy, raw) -> bool:
    return (tidy.exists() and raw.exists()
            and tidy.stat().st_mtime >= raw.stat().st_mtime)


def _stamp_path():
    return RAW / "football-data" / "_stamps.json"


def _stamps() -> dict:
    path = _stamp_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def download_season(code: str, season: str, refresh: bool = False) -> pd.DataFrame | None:
    path = _cache_path(code, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return _read_csv(path)
    url = FOOTBALL_DATA_URL.format(season=season, code=code)
    key = f"{code}_{season}"
    stamps = _stamps()
    headers = {"User-Agent": "Mozilla/5.0"}
    if path.exists() and stamps.get(key):
        headers["If-Modified-Since"] = stamps[key]
    try:
        resp = requests.get(url, timeout=30, headers=headers)
        if resp.status_code == 304 and path.exists():
            return _read_csv(path)
        resp.raise_for_status()
    except requests.RequestException:
        return _read_csv(path) if path.exists() else None
    if resp.headers.get("Last-Modified"):
        stamps[key] = resp.headers["Last-Modified"]
        _stamp_path().write_text(json.dumps(stamps), encoding="utf-8")
    if not resp.content.strip() or b"<html" in resp.content[:200].lower():
        return None
    df = _read_csv(io.BytesIO(resp.content))
    df.to_csv(path, index=False)
    return df


def _parse_dates(raw: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    fallback = pd.to_datetime(raw, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(fallback)


def _best_odds(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index, columns=["odds_h", "odds_d", "odds_a"], dtype=float)
    for h, d, a in ODDS_PREFERENCE:
        if not {h, d, a} <= set(df.columns):
            continue
        trio = df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
        usable = trio.notna().all(axis=1) & out["odds_h"].isna()
        out.loc[usable, ["odds_h", "odds_d", "odds_a"]] = trio.loc[usable].values
    return out


def normalise(df: pd.DataFrame, code: str, season: str) -> pd.DataFrame:
    if df is None or "HomeTeam" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame()
    present = {k: v for k, v in CORE.items() if k in df.columns}
    out = df[list(present)].rename(columns=present).copy()
    out["date"] = _parse_dates(df["Date"].astype(str).str.strip())
    out = out.join(_best_odds(df))
    out["league"] = code
    out["season"] = season_label(season)
    out["season_code"] = season
    out = out.dropna(subset=["date", "home", "away", "hg", "ag"])
    for col in ("hg", "ag"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["hg", "ag"]).reset_index(drop=True)


def _history_path():
    return PROCESSED / "history_football_data.parquet"


def _history_is_stale(codes: list[str], live: str) -> bool:
    path = _history_path()
    if not path.exists():
        return True
    stamp = path.stat().st_mtime
    return any(_cache_path(code, season).stat().st_mtime > stamp
               for code in codes for season in season_codes() if season != live
               and _cache_path(code, season).exists())


def _history(codes: list[str], live: str, refresh: bool, pause: float) -> pd.DataFrame:
    path = _history_path()
    if not refresh and not _history_is_stale(codes, live):
        return pd.read_parquet(path)
    frames = []
    for code in codes:
        for season in season_codes():
            if season == live:
                continue
            raw = download_season(code, season, refresh=refresh)
            if raw is None:
                continue
            tidy = normalise(raw, code, season)
            if not tidy.empty:
                frames.append(tidy)
            if refresh:
                time.sleep(pause)
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not history.empty:
        history.to_parquet(path, index=False)
    return history


def load_all(codes: list[str] | None = None, refresh: bool = False,
             pause: float = 0.3, current_only: bool = False) -> pd.DataFrame:
    codes = codes or list(LEAGUES)
    live = current_season_code()
    frames = [_history(codes, live, refresh, pause)]
    fresh = refresh or current_only
    with ThreadPoolExecutor(max_workers=len(codes)) as pool:
        current = pool.map(lambda c: (c, download_season(c, live, refresh=fresh)), codes)
    for code, raw in current:
        if raw is None:
            continue
        tidy = normalise(raw, code, live)
        if not tidy.empty:
            frames.append(tidy)
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("no seasons downloaded")
    matches = pd.concat(frames, ignore_index=True)
    return matches.sort_values("date").reset_index(drop=True)
