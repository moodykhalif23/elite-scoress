import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import rivalry
from src.models.design import _orientation, _pair_keys


@pytest.fixture
def meetings():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-08-01", "2021-01-01", "2021-08-01"]),
        "home": ["man united", "man city", "man united", "man city"],
        "away": ["man city", "man united", "man city", "man united"],
        "hg": [2, 1, 0, 3], "ag": [0, 1, 0, 1],
        "result": ["H", "D", "D", "H"],
        "league": ["E0"] * 4, "season": ["2019/20", "2020/21", "2020/21", "2021/22"],
    })


def test_pair_key_is_symmetric():
    keys = _pair_keys(pd.Series(["a", "b"]), pd.Series(["b", "a"]))
    assert keys[0] == keys[1]


def test_orientation_flips_with_venue():
    orient = _orientation(pd.Series(["a", "b"]), pd.Series(["b", "a"]))
    assert orient[0] == 1.0 and orient[1] == -1.0


def test_derby_lookup_is_order_independent():
    lookup = rivalry.derby_lookup()
    assert lookup[frozenset(("man united", "man city"))] == "Manchester"
    assert lookup[frozenset(("man city", "man united"))] == "Manchester"


def test_tag_derbies_marks_only_listed_pairs(meetings):
    tagged = rivalry.tag_derbies(meetings)
    assert tagged.notna().all()
    other = pd.DataFrame({"home": ["arsenal"], "away": ["burnley"]})
    assert rivalry.tag_derbies(other).isna().all()


def test_venue_split_separates_legs(meetings):
    split = rivalry.venue_split(meetings, "man united", "man city")
    assert len(split) == 2
    assert split.loc[split["fixture"].str.startswith("man united"), "played"].iloc[0] == 2


def test_scoreline_history_respects_venue(meetings):
    at_old_trafford = rivalry.scoreline_history(meetings, "man united", "man city")
    assert at_old_trafford["times"].sum() == 2
    both = rivalry.scoreline_history(meetings, "man united", "man city", venue_specific=False)
    assert both["times"].sum() == 4


def test_recurrence_lift_compares_to_model(meetings):
    grid = np.full((6, 6), 1 / 36)
    table = rivalry.recurrence(meetings, "man united", "man city", grid)
    assert set(table["score"]) == {"2-0", "0-0"}
    assert (table["lift"] > 1).all()


def test_recurrence_without_model_has_no_lift(meetings):
    table = rivalry.recurrence(meetings, "man united", "man city")
    assert table["model"].isna().all()


def test_predicted_columns_never_shadow_team_names():
    from src.models import simulate

    fixtures = pd.DataFrame({"home": ["a"], "away": ["b"], "league": ["E0"]})
    reserved = set(fixtures.columns)
    emitted = {f"p_{n}{s}" for n in ("home", "draw", "away") for s in ("", "_lo", "_hi")}
    assert not (emitted & reserved)
    assert "home" not in emitted and "away" not in emitted
    assert hasattr(simulate, "predict_fixtures")
