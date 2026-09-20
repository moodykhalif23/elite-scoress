from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.config import PROCESSED
from src.features import build as features
from src.features import players as player_features


@dataclass
class Report:
    previous: int
    total: int
    added: int
    retrained: bool
    seconds: float

    def __str__(self) -> str:
        head = (f"{self.added} new result(s) · {self.total:,} matches total "
                f"({self.seconds:.1f}s)")
        return head + (" · model retrained" if self.retrained else " · model unchanged")


def _existing_count() -> int:
    path = PROCESSED / "matches.parquet"
    if not path.exists():
        return 0
    return len(pd.read_parquet(path, columns=["match_id"]))


def run(retrain: bool = True, variant: str = "static", force: bool = False,
        with_players: bool = True) -> Report:
    started = time.time()
    previous = _existing_count()
    matches = features.build(current_only=True)
    features.save(matches)
    added = len(matches) - previous

    if with_players and (added > 0 or force or previous == 0):
        values = player_features.build_values(current_only=True)
        if not values.empty:
            player_features.save(values)

    did_train = False
    if retrain and (added > 0 or force or previous == 0):
        from src.cli import save_model, select
        from src.models.design import build_design

        model = select(variant)
        design = build_design(matches, decay=variant == "static")
        save_model(model.fit_map(design), design, variant)
        did_train = True

    return Report(previous, len(matches), added, did_train, time.time() - started)
