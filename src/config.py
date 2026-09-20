from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "data" / "artifacts"

FIRST_SEASON = 2000


@dataclass(frozen=True)
class League:
    code: str
    name: str
    country: str
    understat: str


LEAGUES = {
    "E0": League("E0", "Premier League", "England", "EPL"),
    "SP1": League("SP1", "La Liga", "Spain", "La liga"),
    "I1": League("I1", "Serie A", "Italy", "Serie A"),
    "D1": League("D1", "Bundesliga", "Germany", "Bundesliga"),
}

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
UNDERSTAT_URL = "https://understat.com/getLeagueData/{league}/{year}"

HALF_LIFE_DAYS = 550.0
XG_WEIGHT = 0.5
MAX_GOALS = 10


def current_season_start() -> int:
    today = date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_codes(first: int = FIRST_SEASON, last: int | None = None) -> list[str]:
    last = last if last is not None else current_season_start()
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(first, last + 1)]


def season_label(code: str) -> str:
    start = int(code[:2])
    century = 2000 if start < 90 else 1900
    return f"{century + start}/{str(century + start + 1)[2:]}"


for _d in (RAW, PROCESSED, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)
