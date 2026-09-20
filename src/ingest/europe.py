from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.config import PROCESSED, RAW
from src.ingest.teams import canonical

BASE_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master/{season}/{code}.txt"
COMPETITIONS = {"UCL": "cl", "UEL": "el"}
FIRST_SEASON = 2011

COUNTRY_TO_LEAGUE = {"ENG": "E0", "ESP": "SP1", "ITA": "I1", "GER": "D1"}
FIXTURE_URL = "https://fixturedownload.com/download/{slug}-{year}-UTC.csv"
FIXTURE_SLUGS = {"UCL": "champions-league", "UEL": "europa-league"}
DATE_LINE = re.compile(r"^\s{2,}\w{3}\s+(\w{3})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_LINE = re.compile(
    r"^\s+(?:\d{1,2}:\d{2}\s+)?(.+?)\s+\(([A-Z]{3})\)\s+v\s+(.+?)\s+\(([A-Z]{3})\)\s+(.+?)\s*$")
SCORE = re.compile(r"(\d+)-(\d+)")
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def season_label(start: int) -> str:
    return f"{start}-{str(start + 1)[2:]}"


def _fetch(competition: str, start: int, refresh: bool = False) -> str | None:
    code = COMPETITIONS[competition]
    path = RAW / "europe" / f"{code}_{start}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    url = BASE_URL.format(season=season_label(start), code=code)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    path.write_text(resp.text, encoding="utf-8")
    return resp.text


def _regulation_score(tail: str) -> tuple[int, int] | None:
    if "a.e.t." in tail:
        inside = re.search(r"\(([^)]*)\)", tail)
        if not inside:
            return None
        first = SCORE.search(inside.group(1))
        return (int(first.group(1)), int(first.group(2))) if first else None
    head = tail.split("(")[0]
    found = SCORE.search(head)
    return (int(found.group(1)), int(found.group(2))) if found else None


def parse(text: str, competition: str, start: int) -> pd.DataFrame:
    rows, day, month, year = [], None, None, start
    for line in text.splitlines():
        stamp = DATE_LINE.match(line)
        if stamp:
            month, day = MONTHS[stamp.group(1)], int(stamp.group(2))
            if stamp.group(3):
                year = int(stamp.group(3))
            elif month <= 7 and start == year:
                year = start + 1
            continue
        entry = MATCH_LINE.match(line)
        if not entry or day is None:
            continue
        score = _regulation_score(entry.group(5))
        if score is None:
            continue
        rows.append({
            "date": pd.Timestamp(year=year, month=month, day=day),
            "home": canonical(entry.group(1)), "away": canonical(entry.group(3)),
            "home_country": entry.group(2), "away_country": entry.group(4),
            "hg": score[0], "ag": score[1], "competition": competition,
            "season": f"{start}/{str(start + 1)[2:]}",
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["result"] = ["H" if h > a else ("A" if a > h else "D")
                       for h, a in zip(frame["hg"], frame["ag"])]
    frame["league"] = frame["competition"]
    return frame


def load_all(refresh: bool = False, last: int | None = None) -> pd.DataFrame:
    from src.config import current_season_start

    last = last if last is not None else current_season_start()
    jobs = [(comp, year) for comp in COMPETITIONS
            for year in range(FIRST_SEASON, last + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        texts = pool.map(lambda job: (job, _fetch(job[0], job[1], refresh)), jobs)
    frames = [parse(text, comp, year) for (comp, year), text in texts if text]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def _fixture_csv(competition: str, year: int, refresh: bool = True) -> pd.DataFrame:
    path = RAW / "europe" / f"fixtures_{FIXTURE_SLUGS[competition]}_{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        url = FIXTURE_URL.format(slug=FIXTURE_SLUGS[competition], year=year)
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            path.write_bytes(resp.content)
        except requests.RequestException:
            if not path.exists():
                return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _schedule(year: int, refresh: bool = True) -> pd.DataFrame:
    frames = []
    for competition in FIXTURE_SLUGS:
        raw = _fixture_csv(competition, year, refresh)
        if raw.empty or "Home Team" not in raw.columns:
            continue
        frame = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], format="%d/%m/%Y %H:%M", errors="coerce"),
            "home_name": raw["Home Team"], "away_name": raw["Away Team"],
            "home": raw["Home Team"].map(canonical), "away": raw["Away Team"].map(canonical),
            "league": competition, "round": raw.get("Round Number"),
            "score": raw.get("Result"),
        })
        frames.append(frame.dropna(subset=["date", "home", "away"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def upcoming(year: int | None = None, refresh: bool = True) -> pd.DataFrame:
    from src.config import current_season_start

    year = year if year is not None else current_season_start()
    schedule = _schedule(year, refresh)
    if schedule.empty:
        return pd.DataFrame()
    unplayed = schedule["score"].isna() | (schedule["score"].astype(str).str.strip() == "")
    out = schedule[unplayed].drop(columns=["score"]).copy()
    for column in ("odds_h", "odds_d", "odds_a"):
        out[column] = float("nan")
    return out.reset_index(drop=True)


def played(year: int | None = None, refresh: bool = True) -> pd.DataFrame:
    from src.config import current_season_start

    year = year if year is not None else current_season_start()
    schedule = _schedule(year, refresh)
    if schedule.empty:
        return pd.DataFrame()
    scores = schedule["score"].astype(str).str.extract(r"(\d+)\s*-\s*(\d+)")
    frame = schedule.assign(hg=pd.to_numeric(scores[0], errors="coerce"),
                            ag=pd.to_numeric(scores[1], errors="coerce"))
    frame = frame.dropna(subset=["hg", "ag"]).drop(columns=["score", "round"])
    if frame.empty:
        return frame
    frame["result"] = ["H" if h > a else ("A" if a > h else "D")
                       for h, a in zip(frame["hg"], frame["ag"])]
    frame["competition"] = frame["league"]
    frame["season"] = f"{year}/{str(year + 1)[2:]}"
    frame["home_country"] = None
    frame["away_country"] = None
    return frame.drop(columns=["home_name", "away_name"]).reset_index(drop=True)


def save(frame: pd.DataFrame, name: str = "europe.parquet") -> str:
    path = PROCESSED / name
    frame.to_parquet(path, index=False)
    return str(path)
