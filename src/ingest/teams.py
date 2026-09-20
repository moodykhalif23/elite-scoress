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
    "bayern munchen": "bayern munich", "munchen": "bayern munich",
    "monchengladbach": "mgladbach", "bor monchengladbach": "mgladbach",
    "lazio roma": "lazio", "internazionale milano": "inter", "internazionale": "inter",
    "atalanta bc": "atalanta", "leicester city": "leicester",
    "sociedad futbol": "sociedad", "atletico de madrid": "ath madrid",
    "athletic bilbao": "ath bilbao", "sevilla": "sevilla", "porto": "porto",
}


TRANSLITERATE = str.maketrans({"ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l",
                               "æ": "ae", "Æ": "ae", "œ": "oe", "ð": "d", "þ": "th",
                               "ı": "i", "ß": "ss"})


def _fold(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name).translate(TRANSLITERATE))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("'", "").replace(".", " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


EUROPEAN = {
    "sl benfica": "benfica", "sport lisboa e benfica": "benfica",
    "sporting": "sporting cp", "sporting clube portugal": "sporting cp",
    "sporting clube de portugal": "sporting cp",
    "sporting braga": "braga", "sporting clube braga": "braga",
    "sporting clube de braga": "braga", "sc braga": "braga",
    "psv eindhoven": "psv", "galatasaray sk": "galatasaray",
    "fk shakhtar donetsk": "shakhtar donetsk", "shakhtar": "shakhtar donetsk",
    "fk crvena zvezda": "crvena zvezda", "crvena zvezda": "crvena zvezda",
    "gnk dinamo zagreb": "dinamo zagreb", "gnk dinamo": "dinamo zagreb",
    "dinamo": "dinamo zagreb", "feyenoord rotterdam": "feyenoord",
    "racing lens": "lens", "racing club de lens": "lens", "racing club lens": "lens",
    "qarabag agdam fk": "qarabag", "qarabag fk": "qarabag",
    "royale union saint gilloise": "union saint gilloise",
    "union sg": "union saint gilloise", "union": "union saint gilloise",
    "sk slavia praha": "slavia praha", "sk sturm graz": "sturm graz",
    "fk bod glimt": "bodo glimt", "bod glimt": "bodo glimt", "glimt": "bodo glimt",
    "brugge kv": "club brugge", "brugge": "club brugge", "club brugge kv": "club brugge",
    "atleti": "ath madrid", "b dortmund": "dortmund", "leipzig": "rb leipzig",
    "man utd": "man united", "paris": "paris saint germain",
    "psg": "paris saint germain", "paris saint germain": "paris saint germain",
    "olympiacos": "olympiakos piraeus", "olympiakos": "olympiakos piraeus",
    "marseille": "olympique marseille", "lyon": "olympique lyonnais",
    "salzburg": "red bull salzburg", "red bull salzburg": "red bull salzburg",
    "s bratislava": "slovan bratislava", "h beer sheva": "hapoel beer sheva",
    "n e c": "nec nijmegen", "nec": "nec nijmegen",
    "sk slovan bratislava": "slovan bratislava",
    "bayer 04 leverkusen": "leverkusen", "eintracht frankfurt": "ein frankfurt",
}
ALIASES.update(EUROPEAN)


def canonical(name: str) -> str:
    folded = re.sub(r"\s+", " ", _fold(name)).strip()
    if folded in ALIASES:
        return ALIASES[folded]
    tokens = [t for t in folded.split() if t not in DROP_TOKENS and not t.isdigit()]
    stripped = " ".join(tokens) if tokens else folded
    return ALIASES.get(stripped, stripped)


def unmatched(left: set[str], right: set[str]) -> tuple[set[str], set[str]]:
    lc = {canonical(n) for n in left}
    rc = {canonical(n) for n in right}
    return lc - rc, rc - lc
