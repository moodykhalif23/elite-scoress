# Football Predictor

Latent-strength match prediction for the Premier League, La Liga, Serie A and Bundesliga,
trained on every result since 2000.

## What it does

A Bayesian hierarchical Poisson model learns a hidden **attack** and **defence** rating for
every team directly from scorelines — no hand-labelled features. Those ratings produce a full
scoreline distribution for any fixture, which is then read off as 1X2, over/under, BTTS and
correct-score probabilities, each with a credible interval.

Recent matches count more than old ones (exponential time decay, 550-day half-life), the
Dixon–Coles correction fixes the well-known Poisson bias on 0-0/1-0/0-1/1-1, and squad
availability shifts the rates before simulation.

## Data

| Source | Contents | Coverage |
| --- | --- | --- |
| football-data.co.uk | results, shots, cards, corners, closing odds | 2000 → now, all 4 leagues |
| Understat | match xG, player minutes / xG / xA | 2014 → now |
| football-data fixtures feed | upcoming matches + current odds | rolling |

Team names differ between sources, so everything is routed through `src/ingest/teams.py`
before joining.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli build            # download + assemble (first run takes a few minutes)
python -m src.cli train --fast     # MAP fit, seconds; drop --fast for full sampling
python -m src.cli predict          # upcoming fixtures
python -m src.cli backtest         # walk-forward evaluation vs the bookmaker
streamlit run app/main.py          # GUI
```

## Layout

```
src/ingest/      football-data, Understat, fixtures, team-name resolution
src/features/    match dataset assembly, player contribution values
src/models/      hierarchical Poisson model, scoreline simulation
src/backtest.py  walk-forward evaluation
app/main.py      Streamlit GUI
```

## How the squad adjustment works

Each player gets a share of their team's attacking output (xG + xA, recency weighted) and, for
defenders and keepers, a share of defensive minutes. Marking a player unavailable removes that
share net of a replacement level, and the team's rates move accordingly. Historic injury lists
aren't freely available, so this layer applies to upcoming fixtures only — the model itself is
never trained on it.

## Tuning

`HALF_LIFE_DAYS` (550) and the MAP prior scale (0.45) were chosen by grid search on seasons
2015/16–2020/21 and confirmed on a held-out 2021/22–2026/27 window, where they improved log
loss from 0.9958 to 0.9936. The surface has an interior optimum — shorter memory forgets too
fast, longer memory carries dead squads.

## Reading the output

`backtest` reports log loss, RPS, Brier and accuracy for the model and for the bookmaker on the
same matches. The bookmaker is the bar: beating it on log loss is hard and is the only result
that means anything. Accuracy alone is not a useful measure here — always-predict-home clears
45% on its own.

Ratings are identified within a league, not across them. Comparing a Bundesliga attack rating
to a La Liga one is not meaningful; these leagues never play each other in this dataset.
