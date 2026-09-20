# Football Predictor

Latent-strength match prediction for the Premier League, La Liga, Serie A and Bundesliga,
trained on every result since 2000.

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

Team names differ between sources, so everything is routed through `src/ingest/teams.py`
before joining.

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

## How the squad adjustment works

Each player gets a share of their team's attacking output (xG + xA, recency weighted) and, for
defenders and keepers, a share of defensive minutes. Marking a player unavailable removes that
share net of a replacement level, and the team's rates move accordingly.

With `API_FOOTBALL_KEY` set, `python -m src.cli injuries` pulls the current injury list and the
GUI pre-selects those players automatically. Four requests cover all four leagues, so the free
tier is sufficient for this. Historic lineups are *not* free, so this layer applies to upcoming
fixtures only — the model is never trained on it.

## Results

Walk-forward over 15,153 out-of-sample matches, retrained from scratch before each season:

| | Log loss | RPS | Accuracy |
| --- | --- | --- | --- |
| Uniform guess | 1.0986 | — | 33.3% |
| Model | **0.9866** | 0.2014 | 52.4% |
| Bookmaker closing odds | 0.9631 | 0.1938 | 54.3% |

The model has real skill but does not beat the market. Betting its disagreements with the book
returns −8% to −14%, against a 5.2% overround — and it gets *worse* as the claimed edge grows,
which is the opposite of what a genuine edge looks like. **Treat this as an analysis tool, not a
staking system.**

## Tuning

Everything below was chosen on seasons 2015/16–2020/21 and confirmed on a held-out
2021/22–2026/27 window, with a paired bootstrap over per-match log losses to check the
differences are real:

| Change | Held-out log loss | Verdict |
| --- | --- | --- |
| baseline (365-day decay, σ=0.30, goals) | 0.9958 | — |
| 550-day half-life, σ=0.45 | 0.9936 | adopted |
| + 50/50 goals/xG blend | 0.9918 | adopted, significant |
| + AR(1) dynamic ratings | 0.9910 | **not** significant vs the line above |

The AR(1) variant's edge over static+xG is −0.0009 with a 95% interval of [−0.0028, +0.0011],
so it is kept available but is not the default. Its fitted persistence (φ=0.99) is close to a
random walk, which says team strength moves slowly — and explains why exponential decay does
almost as well.

## Reading the output

`backtest` reports log loss, RPS, Brier and accuracy for the model and for the bookmaker on the
same matches. The bookmaker is the bar: beating it on log loss is hard and is the only result
that means anything. Accuracy alone is not a useful measure here — always-predict-home clears
45% on its own.

Ratings are identified within a league, not across them. Comparing a Bundesliga attack rating
to a La Liga one is not meaningful; these leagues never play each other in this dataset.
