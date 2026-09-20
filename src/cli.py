from __future__ import annotations

import argparse
import pickle

import pandas as pd

from src.config import ARTIFACTS
from src.features import build as features
from src.features import availability
from src.features import players as player_features
from src.ingest import api_football
from src.ingest import fixtures as fixtures_ingest

def model_path(variant: str = "static"):
    return ARTIFACTS / f"model_{variant}.pkl"


def select(name: str):
    from src.models import dynamic, hierarchical
    return {"static": hierarchical, "dynamic": dynamic}[name]


def save_model(result, design, variant: str) -> str:
    path = model_path(variant)
    with path.open("wb") as fh:
        pickle.dump({"result": result, "design": design, "variant": variant}, fh)
    return str(path)


def load_model(variant: str = "static"):
    path = model_path(variant)
    if not path.exists():
        raise FileNotFoundError(f"no {variant} model; run "
                                f"`python -m src.cli train --model {variant} --fast` first")
    with path.open("rb") as fh:
        blob = pickle.load(fh)
    return blob["result"], blob["design"], blob.get("variant", variant)


def cmd_build(args):
    matches = features.build(refresh=args.refresh, with_xg=not args.no_xg)
    print(features.coverage(matches).to_string(index=False))
    print(f"\ntotal: {len(matches):,} matches | saved:", features.save(matches))
    values = player_features.build_values(refresh=args.refresh)
    if not values.empty:
        print(f"player values: {len(values):,} rows | saved:", player_features.save(values))


def cmd_train(args):
    from src.models import hierarchical
    from src.models.design import build_design

    model = select(args.model)
    matches = features.load()
    design = build_design(matches, decay=args.model == "static")
    print(f"design: {len(design.hg):,} matches | {len(design.teams)} teams | "
          f"{design.n_periods} periods | model={args.model}")
    result = model.fit_map(design) if args.fast or args.model == "dynamic" else \
        model.fit(design, draws=args.draws, tune=args.tune, chains=args.chains)
    print("saved:", save_model(result, design, args.model))
    print(hierarchical.ratings(result, design).head(12).to_string(index=False))


def cmd_predict(args):
    from src.models import simulate

    result, design, _ = load_model()
    upcoming = fixtures_ingest.upcoming()
    if upcoming.empty:
        print("no upcoming fixtures available")
        return
    preds = simulate.predict_fixtures(result, design, upcoming)
    cols = ["date", "league", "home_name", "away_name", "home", "draw", "away",
            "exp_hg", "exp_ag", "top_score", "over_2.5", "btts"]
    show = preds[[c for c in cols if c in preds.columns]].copy()
    for c in ("home", "draw", "away", "over_2.5", "btts"):
        if c in show:
            show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    preds.to_parquet(ARTIFACTS / "predictions.parquet", index=False)


def cmd_injuries(args):
    if not api_football.available():
        print("set API_FOOTBALL_KEY to enable injury pulls (free tier is enough: "
              "4 requests covers all four leagues)")
        return
    reported = availability.current_injuries(budget_limit=args.budget)
    if reported.empty:
        print("no injuries reported")
        return
    values = player_features.load()
    matched = sum(len(availability.unavailable_for(reported, values, team))
                  for team in reported["team"].unique())
    print(f"{len(reported)} reported across {reported['team'].nunique()} teams | "
          f"{matched} matched to rated players")
    print("saved:", availability.save(reported))


def cmd_backtest(args):
    from src import backtest

    matches = features.load()
    preds, summary = backtest.walk_forward(matches, start_season=args.start,
                                           variant=args.model)
    if not summary:
        print("backtest produced no folds")
        return
    print(f"tested matches: {summary['n']:,}")
    for key in ("model", "model_on_odds_subset", "bookmaker"):
        if key in summary:
            row = summary[key]
            print(f"{key:22} log_loss={row['log_loss']:.4f} rps={row['rps']:.4f} "
                  f"brier={row['brier']:.4f} acc={row['accuracy']:.1%}")
    print("saved:", backtest.save(preds))


def main():
    parser = argparse.ArgumentParser(prog="predict")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="download and assemble the dataset")
    b.add_argument("--refresh", action="store_true")
    b.add_argument("--no-xg", action="store_true")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="fit the latent strength model")
    t.add_argument("--model", choices=("static", "dynamic"), default="static")
    t.add_argument("--fast", action="store_true", help="MAP instead of full sampling")
    t.add_argument("--draws", type=int, default=1000)
    t.add_argument("--tune", type=int, default=1000)
    t.add_argument("--chains", type=int, default=4)
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict upcoming fixtures")
    p.set_defaults(func=cmd_predict)

    j = sub.add_parser("injuries", help="pull current injuries (needs API_FOOTBALL_KEY)")
    j.add_argument("--budget", type=int, default=8, help="max API requests to spend")
    j.set_defaults(func=cmd_injuries)

    k = sub.add_parser("backtest", help="walk-forward evaluation against bookmaker odds")
    k.add_argument("--start", default="2015/16")
    k.add_argument("--model", choices=("static", "dynamic"), default="static")
    k.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
