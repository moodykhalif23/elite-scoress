# Elite Scores

Latent-strength match prediction for the Premier League, La Liga, Serie A and Bundesliga,
trained on every result since 2000 — plus the Champions League and Europa League, which put
every club on one cross-league scale.

## What it does

A hierarchical Poisson model learns a hidden **attack** and **defence** rating for every team
directly from match outcomes — no hand-labelled features. Those ratings produce a full scoreline
distribution for any fixture, read off as 1X2, over/under, BTTS and correct-score probabilities.

The fit uses **Poisson pseudo-likelihood** (`y·log λ − λ`), which is valid for continuous
non-negative targets and identical to Poisson for integers. That lets the training target be a
blend of goals and xG rather than goals alone. Recent matches count more (exponential decay,
550-day half-life), and the Dixon–Coles correction fixes the Poisson bias on 0-0/1-0/0-1/1-1.

Two model variants share that likelihood:

- **static** (default) — one rating per team, exponential time decay.
- **dynamic** (`--model dynamic`) — ratings follow an AR(1) path across seasons, so a team's
  strength is a trajectory rather than one blended average. Equal to the static model on
  prediction (see below); better for reading how a side has actually moved.

## Data

| Source | Contents | Coverage |
| --- | --- | --- |
| football-data.co.uk | results, shots, cards, corners, closing odds | 2000 → now, all 4 leagues |
| Understat | match xG, player minutes / xG / xA | 2014 → now |
| football-data fixtures feed | upcoming matches + current odds | rolling |
| openfootball | Champions League + Europa League results | 2011/12 → now |

Team names differ between sources, so everything is routed through `src/ingest/teams.py`
before joining.

## Keeping it current

```bash
python -m src.cli update     
## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.cli build                  # download + assemble (first run takes a few minutes)
python -m src.cli train --fast           # MAP fit, seconds
python -m src.cli train --model dynamic  # AR(1) ratings, needed for the trajectory view
python -m src.cli predict                # upcoming fixtures
python -m src.cli backtest               # walk-forward evaluation vs the bookmaker
python -m src.cli injuries               # current injuries (needs API_FOOTBALL_KEY)
streamlit run app/main.py                # GUI
```

## Layout

```
src/ingest/      football-data, Understat, fixtures, API-Football, team-name resolution
src/features/    match dataset assembly, player values, injury resolution
src/models/      design, static + dynamic models, scoreline simulation
src/evaluate.py  vectorised scoring and walk-forward engine
src/backtest.py  bookmaker comparison and edge tables
scripts/         hyperparameter sweeps and model comparison
app/main.py      Streamlit GUI
```


## Reading the output

`backtest` reports log loss, RPS, Brier and accuracy for the model and for the bookmaker on the
same matches. The bookmaker is the bar: beating it on log loss is hard and is the only result
that means anything. Accuracy alone is not a useful measure here — always-predict-home clears
45% on its own.

Ratings are now identified **across** leagues, because Champions League and Europa League results
connect them. Before those were added, a Bundesliga rating and a La Liga one were not comparable;
they are now, though a club whose only matches are European carries far more uncertainty than its
point estimate suggests.
