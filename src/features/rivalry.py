from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

DERBIES = {
    "E0": [
        ("man united", "man city", "Manchester"), ("liverpool", "everton", "Merseyside"),
        ("arsenal", "tottenham", "North London"), ("chelsea", "fulham", "West London"),
        ("arsenal", "chelsea", "London"), ("chelsea", "tottenham", "London"),
        ("west ham", "tottenham", "London"), ("arsenal", "west ham", "London"),
        ("newcastle", "sunderland", "Tyne-Wear"), ("aston villa", "birmingham", "Second City"),
        ("wolves", "west brom", "Black Country"), ("crystal palace", "brighton", "M23"),
        ("liverpool", "man united", "North West"), ("nottm forest", "leicester", "East Midlands"),
        ("stoke", "west brom", "Midlands"), ("crystal palace", "chelsea", "London"),
    ],
    "SP1": [
        ("madrid", "ath madrid", "Madrid"), ("madrid", "barcelona", "El Clasico"),
        ("barcelona", "espanol", "Barcelona"), ("sevilla", "betis", "Seville"),
        ("ath bilbao", "sociedad", "Basque"), ("celta", "la coruna", "Galician"),
        ("valencia", "levante", "Valencia"), ("madrid", "vallecano", "Madrid"),
        ("ath madrid", "vallecano", "Madrid"), ("getafe", "leganes", "Madrid"),
        ("ath madrid", "getafe", "Madrid"), ("ath bilbao", "alaves", "Basque"),
        ("sevilla", "madrid", "Clasico-adjacent"), ("valencia", "villarreal", "Valencian"),
    ],
    "I1": [
        ("inter", "milan", "Milano"), ("roma", "lazio", "Capitale"),
        ("juventus", "torino", "Torino"), ("genoa", "sampdoria", "Genoa"),
        ("juventus", "inter", "Derby d'Italia"), ("napoli", "roma", "Sud"),
        ("fiorentina", "juventus", "Fiorentina-Juve"), ("bologna", "fiorentina", "Apennine"),
        ("verona", "chievo", "Verona"), ("lecce", "bari", "Puglia"),
        ("milan", "juventus", "Grande"), ("atalanta", "brescia", "Lombardy"),
        ("palermo", "catania", "Sicilian"), ("cagliari", "sassuolo", "-"),
    ],
    "D1": [
        ("dortmund", "schalke", "Revierderby"), ("bayern munich", "nurnberg", "Bavarian"),
        ("hamburg", "st pauli", "Hamburg"), ("koln", "mgladbach", "Rhineland"),
        ("koln", "leverkusen", "Rhineland"), ("mgladbach", "leverkusen", "Rhineland"),
        ("hertha", "union berlin", "Berlin"), ("werder bremen", "hamburg", "Nordderby"),
        ("bayern munich", "dortmund", "Der Klassiker"),
        ("stuttgart", "karlsruhe", "Baden-Wurttemberg"),
        ("bayern munich", "1860 munich", "Munich"), ("schalke", "bochum", "Revier"),
        ("hannover", "braunschweig", "Lower Saxony"), ("mainz", "ein frankfurt", "Rhine-Main"),
    ],
}


def derby_lookup() -> dict[frozenset, str]:
    return {frozenset((home, away)): name
            for pairs in DERBIES.values() for home, away, name in pairs}


def tag_derbies(matches: pd.DataFrame) -> pd.Series:
    lookup = derby_lookup()
    keys = [frozenset((h, a)) for h, a in zip(matches["home"], matches["away"])]
    return pd.Series([lookup.get(k) for k in keys], index=matches.index, name="derby")


def pair_key(home: pd.Series, away: pd.Series) -> pd.Series:
    return pd.Series([" v ".join(sorted((h, a))) for h, a in zip(home, away)],
                     index=home.index)


def head_to_head(matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    mask = ((matches["home"] == home) & (matches["away"] == away)) | \
           ((matches["home"] == away) & (matches["away"] == home))
    return matches[mask].sort_values("date", ascending=False)


def venue_split(matches: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    history = head_to_head(matches, home, away)
    rows = []
    for label, host, guest in ((f"{home} at home", home, away), (f"{away} at home", away, home)):
        leg = history[(history["home"] == host) & (history["away"] == guest)]
        if leg.empty:
            continue
        rows.append({
            "fixture": label, "played": len(leg),
            "host_wins": int((leg["result"] == "H").sum()),
            "draws": int((leg["result"] == "D").sum()),
            "guest_wins": int((leg["result"] == "A").sum()),
            "host_goals": round(leg["hg"].mean(), 2),
            "guest_goals": round(leg["ag"].mean(), 2),
        })
    return pd.DataFrame(rows)


def scoreline_history(matches: pd.DataFrame, home: str, away: str,
                      venue_specific: bool = True) -> pd.DataFrame:
    history = head_to_head(matches, home, away)
    if venue_specific:
        history = history[(history["home"] == home) & (history["away"] == away)]
    if history.empty:
        return pd.DataFrame(columns=["score", "times", "share"])
    scores = (history["hg"].astype(int).astype(str) + "-"
              + history["ag"].astype(int).astype(str))
    counts = scores.value_counts()
    return pd.DataFrame({"score": counts.index, "times": counts.to_numpy(),
                         "share": (counts / len(history)).to_numpy()})


def recurrence(matches: pd.DataFrame, home: str, away: str, grid: np.ndarray | None = None,
               venue_specific: bool = True) -> pd.DataFrame:
    history = scoreline_history(matches, home, away, venue_specific)
    if history.empty or grid is None:
        if not history.empty:
            history["model"] = np.nan
        return history

    played = int(history["times"].sum())
    modelled, surprise = [], []
    for score, times in zip(history["score"], history["times"]):
        h, a = (int(part) for part in score.split("-"))
        expected = grid[h, a] if h < grid.shape[0] and a < grid.shape[1] else 0.0
        modelled.append(expected)
        if expected <= 0:
            surprise.append(np.nan)
            continue
        surprise.append(binomtest(int(times), played, expected,
                                  alternative="greater").pvalue)
    history["model"] = modelled
    history["lift"] = history["share"] / history["model"].replace(0, np.nan)
    history["p_value"] = surprise
    history["beyond_chance"] = history["p_value"] < 0.05
    return history
