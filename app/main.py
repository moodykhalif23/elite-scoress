from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS, LEAGUES
from src.features import build as features
from src.features import rivalry
from src.features import availability
from src.features import players as player_features
from src.ingest import fixtures as fixtures_ingest
from src.models import hierarchical, simulate

st.set_page_config(page_title="Football Predictor", page_icon="⚽", layout="wide")


@st.cache_resource
def get_model(variant: str = "static"):
    from src.cli import load_model
    return load_model(variant)


@st.cache_data
def get_matches():
    return features.load()


@st.cache_data
def get_values():
    return player_features.load()


@st.cache_data(ttl=3600)
def get_injuries():
    return availability.load()


@st.cache_data(ttl=3600)
def get_fixtures():
    return fixtures_ingest.upcoming()


def outcome_bar(row) -> go.Figure:
    fig = go.Figure()
    for label, key, colour in [("Home", "p_home", "#2E86DE"), ("Draw", "p_draw", "#8395A7"),
                               ("Away", "p_away", "#EE5A24")]:
        fig.add_bar(x=[row[key] * 100], y=[""], orientation="h", name=label,
                    marker_color=colour, text=f"{label} {row[key]*100:.0f}%",
                    textposition="inside", insidetextanchor="middle")
    fig.update_layout(barmode="stack", height=90, showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=10),
                      xaxis=dict(range=[0, 100], visible=False), yaxis=dict(visible=False))
    return fig


def scoreline_heatmap(grid: np.ndarray, home: str, away: str, top: int = 6) -> go.Figure:
    sub = grid[:top, :top] * 100
    fig = px.imshow(sub, labels=dict(x=f"{away} goals", y=f"{home} goals", color="%"),
                    x=list(range(top)), y=list(range(top)),
                    color_continuous_scale="Blues", text_auto=".1f", aspect="auto")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=20, b=0))
    return fig


def match_detail(result, design, values, matches, fx, injuries=None):
    st.markdown(f"#### {fx['home_name']} vs {fx['away_name']}")
    out_home, out_away = [], []
    if values.empty:
        st.caption("No player data yet — run `python -m src.cli build` to enable squad adjustments.")
    else:
        reported = injuries if injuries is not None else pd.DataFrame()
        left, right = st.columns(2)
        for col, team, name, bucket in [(left, fx["home"], fx["home_name"], "home"),
                                        (right, fx["away"], fx["away_name"], "away")]:
            squad = values[values["team"] == team].sort_values("att_share", ascending=False)
            options = squad["player"].tolist()[:30]
            flagged = availability.unavailable_for(reported, values, team)
            picked = col.multiselect(f"{name} — unavailable", options,
                                     default=[p for p in flagged if p in options],
                                     key=f"{fx['home']}_{fx['away']}_{bucket}")
            (out_home if bucket == "home" else out_away).extend(picked)

    h_att, h_def = player_features.team_adjustment(values, fx["home"], out_home)
    a_att, a_def = player_features.team_adjustment(values, fx["away"], out_away)
    adjust = (h_att + a_def, a_att + h_def)

    pred = simulate.predict_match(result, design, fx["home"], fx["away"], fx["league"], adjust)
    if pred is None:
        st.warning("One of these teams has no rating yet (newly promoted or unmatched name).")
        return

    cols = st.columns(4)
    cols[0].metric("Home win", f"{pred['home']*100:.1f}%",
                   f"{pred['home_lo']*100:.0f}–{pred['home_hi']*100:.0f}% band")
    cols[1].metric("Draw", f"{pred['draw']*100:.1f}%")
    cols[2].metric("Away win", f"{pred['away']*100:.1f}%")
    cols[3].metric("Expected goals", f"{pred['exp_hg']:.2f} – {pred['exp_ag']:.2f}")

    cols = st.columns(4)
    cols[0].metric("Most likely score", pred["top_score"], f"{pred['top_score_p']*100:.1f}%")
    cols[1].metric("Over 2.5", f"{pred['over_2.5']*100:.1f}%")
    cols[2].metric("BTTS", f"{pred['btts']*100:.1f}%")
    if pd.notna(fx.get("odds_h")):
        book = 1 / np.array([fx["odds_h"], fx["odds_d"], fx["odds_a"]])
        book = book / book.sum()
        edge = max(pred["home"] - book[0], pred["draw"] - book[1], pred["away"] - book[2])
        cols[3].metric("Best edge vs book", f"{edge*100:+.1f} pts")

    st.plotly_chart(scoreline_heatmap(pred["scoreline_grid"], fx["home_name"],
                                      fx["away_name"]), width='stretch')
    show_form(matches, fx)
    show_h2h(matches, fx)


def recent_form(matches: pd.DataFrame, team: str, limit: int = 6) -> pd.DataFrame:
    played = matches[(matches["home"] == team) | (matches["away"] == team)]
    played = played.sort_values("date", ascending=False).head(limit)
    rows = []
    for _, m in played.iterrows():
        at_home = m["home"] == team
        gf, ga = (m["hg"], m["ag"]) if at_home else (m["ag"], m["hg"])
        outcome = "W" if gf > ga else ("D" if gf == ga else "L")
        rows.append({"Date": m["date"].date(), "Venue": "H" if at_home else "A",
                     "Opponent": m["away"] if at_home else m["home"],
                     "Score": f"{int(gf)}–{int(ga)}", "Result": outcome})
    return pd.DataFrame(rows)


def form_summary(matches: pd.DataFrame, team: str, limit: int = 6) -> str:
    played = matches[(matches["home"] == team) | (matches["away"] == team)]
    played = played.sort_values("date", ascending=False).head(limit)
    if played.empty:
        return "no recent matches"
    gf = np.where(played["home"] == team, played["hg"], played["ag"])
    ga = np.where(played["home"] == team, played["ag"], played["hg"])
    record = f"{int((gf > ga).sum())}W {int((gf == ga).sum())}D {int((gf < ga).sum())}L"
    return f"{record} · {gf.sum():.0f}-{ga.sum():.0f} goals in last {len(played)}"


def show_form(matches: pd.DataFrame, fx):
    left, right = st.columns(2)
    for col, team, name in ((left, fx["home"], fx["home_name"]),
                            (right, fx["away"], fx["away_name"])):
        col.caption(f"{name} — {form_summary(matches, team)}")
        recent = recent_form(matches, team)
        if not recent.empty:
            col.dataframe(recent, hide_index=True, width='stretch')


def show_h2h(matches: pd.DataFrame, fx, limit: int = 8):
    mask = ((matches["home"] == fx["home"]) & (matches["away"] == fx["away"])) | \
           ((matches["home"] == fx["away"]) & (matches["away"] == fx["home"]))
    h2h = matches[mask].sort_values("date", ascending=False).head(limit)
    if h2h.empty:
        st.caption("No recorded head-to-head since 2000.")
        return
    table = pd.DataFrame({
        "Date": h2h["date"].dt.date,
        "Match": h2h["home"] + " " + h2h["hg"].astype(int).astype(str) + "–"
                 + h2h["ag"].astype(int).astype(str) + " " + h2h["away"],
        "Season": h2h["season"],
    })
    st.caption("Head-to-head")
    st.dataframe(table, hide_index=True, width='stretch')


def page_fixtures(result, design, values, matches):
    fixtures = get_fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures in the feed right now.")
        return
    names = {code: lg.name for code, lg in LEAGUES.items()}
    chosen = st.sidebar.multiselect("Leagues", list(names), default=list(names),
                                    format_func=lambda c: names[c])
    fixtures = fixtures[fixtures["league"].isin(chosen)]
    if fixtures.empty:
        st.info("No fixtures for the selected leagues.")
        return

    preds = simulate.predict_fixtures(result, design, fixtures)
    if preds.empty:
        st.warning("No fixtures could be matched to rated teams.")
        return

    st.subheader(f"{len(preds)} upcoming fixtures")
    for _, row in preds.iterrows():
        header = (f"{row['date'].date()}  ·  {names.get(row['league'], row['league'])}  ·  "
                  f"{row['home_name']} vs {row['away_name']}  —  "
                  f"{row['p_home']*100:.0f}/{row['p_draw']*100:.0f}/{row['p_away']*100:.0f}")
        with st.expander(header):
            st.plotly_chart(outcome_bar(row), width='stretch')
            match_detail(result, design, values, matches, row, get_injuries())


def page_ratings(result, design):
    table = hierarchical.ratings(result, design)
    names = {code: lg.name for code, lg in LEAGUES.items()}
    table["league_name"] = table["league"].map(names)
    league = st.sidebar.selectbox("League", sorted(table["league_name"].dropna().unique()))
    sub = table[table["league_name"] == league]
    fig = px.scatter(sub, x="attack", y="defence", text="team", color="strength",
                     color_continuous_scale="RdYlBu", height=620,
                     labels={"attack": "Attack (higher scores more)",
                             "defence": "Defence (higher concedes less)"})
    fig.update_traces(textposition="top center", marker=dict(size=11))
    fig.add_hline(y=0, line_dash="dot", opacity=0.3)
    fig.add_vline(x=0, line_dash="dot", opacity=0.3)
    st.plotly_chart(fig, width='stretch')
    st.caption("Ratings are identified within a league — cross-league values are not comparable.")
    st.dataframe(sub.drop(columns="league_name").round(3), hide_index=True,
                 width='stretch')


def page_trajectory():
    try:
        params, design, _ = get_model("dynamic")
    except FileNotFoundError:
        st.info("Season-by-season ratings need the dynamic model.")
        st.code("python -m src.cli train --model dynamic")
        return
    from src.models import dynamic

    names = {code: lg.name for code, lg in LEAGUES.items()}
    league = st.sidebar.selectbox("League", sorted(names), format_func=lambda c: names[c])
    members = [t for i, t in enumerate(design.teams)
               if design.leagues[int(np.argmax(design.membership[i]))] == league]
    chosen = st.sidebar.multiselect("Teams", members, default=members[:4])
    if not chosen:
        st.info("Pick at least one team.")
        return

    frames = []
    for team in chosen:
        path = dynamic.trajectory(params, design, team)
        path["team"] = team
        frames.append(path)
    history = pd.concat(frames, ignore_index=True)
    history["strength"] = history["attack"] + history["defence"]

    metric = st.radio("Series", ["strength", "attack", "defence"], horizontal=True)
    fig = px.line(history, x="season", y=metric, color="team", markers=True, height=520)
    fig.add_hline(y=0, line_dash="dot", opacity=0.3)
    fig.update_layout(xaxis_title="", yaxis_title=metric.title())
    st.plotly_chart(fig, width='stretch')
    st.caption("Ratings follow an AR(1) path across seasons, so a team's line reflects "
               "how its strength actually moved rather than one blended average.")


def page_rivalry(result, design, matches):
    names = {code: lg.name for code, lg in LEAGUES.items()}
    league = st.sidebar.selectbox("League", sorted(names), format_func=lambda c: names[c])
    pairs = rivalry.DERBIES.get(league, [])
    labels = [f"{h} v {a}  ·  {tag}" for h, a, tag in pairs]
    picked = st.sidebar.selectbox("Fixture", range(len(labels)),
                                  format_func=lambda i: labels[i]) if labels else None
    if picked is None:
        st.info("No derbies listed for this league.")
        return
    home, away, tag = pairs[picked]

    history = rivalry.head_to_head(matches, home, away)
    if history.empty:
        st.warning(f"No meetings between {home} and {away} since 2000.")
        return

    st.subheader(f"{home} v {away}")
    st.caption(f"{tag} derby · {len(history)} meetings since 2000")

    split = rivalry.venue_split(matches, home, away)
    if not split.empty:
        st.dataframe(split, hide_index=True, width='stretch')

    pred = simulate.predict_match(result, design, home, away, league)
    cols = st.columns(4)
    if pred:
        cols[0].metric(f"{home} win", f"{pred['home']*100:.1f}%")
        cols[1].metric("Draw", f"{pred['draw']*100:.1f}%")
        cols[2].metric(f"{away} win", f"{pred['away']*100:.1f}%")
        cols[3].metric("Expected goals", f"{pred['exp_hg']:.2f} – {pred['exp_ag']:.2f}")

    venue_only = st.checkbox(f"Only when {home} host", value=True)
    grid = pred["scoreline_grid"] if pred else None
    table = rivalry.recurrence(matches, home, away, grid, venue_specific=venue_only)
    if table.empty:
        st.info("No meetings at this venue.")
        return

    st.markdown("#### Which scorelines actually recur")
    show = table.head(10).copy()
    show["share"] = (show["share"] * 100).round(1)
    if "model" in show:
        show["model"] = (show["model"] * 100).round(1)
    if "lift" in show:
        show["lift"] = show["lift"].round(2)
    columns = {"score": "Score", "times": "Times", "share": "Historic %",
               "model": "Model %", "lift": "Lift"}
    st.dataframe(show[[c for c in columns if c in show]].rename(columns=columns),
                 hide_index=True, width='stretch')
    st.caption(
        "Lift above 1 means the scoreline has come up more often here than the model expects. "
        "Across all 57 listed derbies, 52 scorelines clear a naive p<0.05 test against roughly "
        "26 expected by chance — and **none survive correction for multiple testing**. Treat "
        "every apparent pattern here as description, not prediction.")

    if grid is not None:
        st.plotly_chart(scoreline_heatmap(grid, home, away), width='stretch')


def page_backtest():
    path = ARTIFACTS / "backtest.parquet"
    if not path.exists():
        st.info("Run `python -m src.cli backtest` to generate evaluation results.")
        return
    from src.backtest import OUTCOMES, implied_probabilities

    preds = pd.read_parquet(path)
    actual = preds["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    probs = np.asarray(preds[OUTCOMES], dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)

    cols = st.columns(3)
    cols[0].metric("Matches tested", f"{len(preds):,}")
    cols[1].metric("Accuracy", f"{(probs.argmax(axis=1) == actual).mean():.1%}")
    cols[2].metric("Log loss", f"{-np.log(probs[np.arange(len(actual)), actual]).mean():.4f}")

    mask = preds[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
    if mask.sum() > 50:
        book = implied_probabilities(preds[mask])
        b_loss = -np.log(book[np.arange(mask.sum()), actual[mask.to_numpy()]]).mean()
        st.caption(f"Bookmaker log loss on the same matches: {b_loss:.4f} — "
                   "lower is better, the book is the bar to beat.")

    bins = np.linspace(0, 1, 11)
    flat_p = probs.ravel()
    flat_a = np.zeros_like(probs)
    flat_a[np.arange(len(actual)), actual] = 1
    frame = pd.DataFrame({"p": flat_p, "hit": flat_a.ravel()})
    frame["bin"] = pd.cut(frame["p"], bins)
    cal = frame.groupby("bin", observed=True).agg(predicted=("p", "mean"),
                                                  observed=("hit", "mean"),
                                                  n=("hit", "size")).dropna()
    fig = px.scatter(cal, x="predicted", y="observed", size="n", height=460,
                     labels={"predicted": "Predicted probability",
                             "observed": "Observed frequency"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot"))
    st.plotly_chart(fig, width='stretch')


def main():
    st.title("⚽ Football Predictor")
    try:
        result, design, _ = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m src.cli build\npython -m src.cli train --fast")
        return

    page = st.sidebar.radio("View", ["Fixtures", "Team ratings", "Derbies & H2H",
                                     "Trajectory", "Backtest"])
    if page == "Fixtures":
        page_fixtures(result, design, get_values(), get_matches())
    elif page == "Team ratings":
        page_ratings(result, design)
    elif page == "Derbies & H2H":
        page_rivalry(result, design, get_matches())
    elif page == "Trajectory":
        page_trajectory()
    else:
        page_backtest()


if __name__ == "__main__":
    main()
