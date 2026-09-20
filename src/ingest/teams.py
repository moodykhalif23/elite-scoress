from __future__ import annotations

import re
import unicodedata

DROP_TOKENS = {
    "fc", "cf", "afc", "ac", "as", "ss", "ssc", "us", "usc", "sc", "sv", "tsg", "vfb", "vfl",
    "fsv", "sg", "spvgg", "bsc", "1", "04", "05", "96", "98", "1899", "1900", "de", "calcio",
    "club", "cd", "ud", "rcd", "rc", "sd", "cp", "deportivo", "real", "borussia", "bayer",
    "eintracht", "hamburger", "rasenballsport", "queens", "park",
}

ALIASES = {
    "manchester united": "man united", "manchester city": "man city",
    "newcastle united": "newcastle", "wolverhampton wanderers": "wolves",
    "nottingham forest": "nottm forest", "west bromwich albion": "west brom",
    "queens park rangers": "qpr", "sheffield united": "sheffield united",
    "tottenham hotspur": "tottenham", "brighton and hove albion": "brighton",
    "west ham united": "west ham", "leeds united": "leeds",
    "atletico madrid": "ath madrid", "athletic club": "ath bilbao",
    "espanyol": "espanol", "real sociedad": "sociedad", "real betis": "betis",
    "rayo vallecano": "vallecano", "deportivo la coruna": "la coruna",
    "celta vigo": "celta", "sporting gijon": "sp gijon", "racing santander": "santander",
    "real valladolid": "valladolid", "real zaragoza": "zaragoza",
    "ac milan": "milan", "internazionale": "inter", "hellas verona": "verona",
    "spal 2013": "spal", "chievo verona": "chievo",
    "borussia dortmund": "dortmund", "borussia m gladbach": "mgladbach",
    "borussia monchengladbach": "mgladbach", "bayer leverkusen": "leverkusen",
    "eintracht frankfurt": "ein frankfurt", "fc cologne": "koln", "fc koln": "koln",
    "cologne": "koln", "hamburger sv": "hamburg", "hannover 96": "hannover",
    "mainz 05": "mainz", "schalke 04": "schalke", "hertha berlin": "hertha",
    "rasenballsport leipzig": "rb leipzig", "arminia bielefeld": "bielefeld",
    "fortuna duesseldorf": "fortuna dusseldorf", "greuther fuerth": "greuther furth",
    "nuernberg": "nurnberg", "st pauli": "st pauli", "bayern munich": "bayern munich",
}


def _fold(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("'", "").replace(".", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def canonical(name: str) -> str:
    folded = re.sub(r"\s+", " ", _fold(name)).strip()
    if folded in ALIASES:
        return ALIASES[folded]
    tokens = [t for t in folded.split() if t not in DROP_TOKENS]
    stripped = " ".join(tokens) if tokens else folded
    return ALIASES.get(stripped, stripped)


def unmatched(left: set[str], right: set[str]) -> tuple[set[str], set[str]]:
    lc = {canonical(n) for n in left}
    rc = {canonical(n) for n in right}
    return lc - rc, rc - lc
