from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import theme
from src.config import ARTIFACTS, LEAGUES
from src.features import availability, build as features, players as player_features, rivalry
from src.ingest import fixtures as fixtures_ingest
from src.models import hierarchical, simulate

st.set_page_config(page_title="Elite Scores", page_icon="◍", layout="wide")
LEAGUE_NAMES = {code: lg.name for code, lg in LEAGUES.items()}
COMPETITIONS = {**LEAGUE_NAMES, "UCL": "Champions League", "UEL": "Europa League"}


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


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="fp-eyebrow">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="fp-note">{text}</div>', unsafe_allow_html=True)


def league_filter(key: str, options: list[str] | None = None) -> list[str]:
    options = options or list(COMPETITIONS)
    chosen = st.pills("Competitions", options, default=options,
                      format_func=lambda c: COMPETITIONS.get(c, c),
                      selection_mode="multi", key=key)
    return chosen or options


def form_results(matches: pd.DataFrame, team: str, limit: int = 6) -> list[str]:
    played = matches[(matches["home"] == team) | (matches["away"] == team)]
    played = played.sort_values("date", ascending=False).head(limit)
    gf = np.where(played["home"] == team, played["hg"], played["ag"])
    ga = np.where(played["home"] == team, played["ag"], played["hg"])
    return ["W" if a > b else ("D" if a == b else "L") for a, b in zip(gf, ga)][::-1]


def recent_matches(matches: pd.DataFrame, team: str, limit: int = 6) -> pd.DataFrame:
    played = matches[(matches["home"] == team) | (matches["away"] == team)]
    played = played.sort_values("date", ascending=False).head(limit)
    rows = []
    for _, m in played.iterrows():
        at_home = m["home"] == team
        gf, ga = (m["hg"], m["ag"]) if at_home else (m["ag"], m["hg"])
        rows.append({"Date": m["date"].date(), "": "H" if at_home else "A",
                     "Opponent": (m["away"] if at_home else m["home"]).title(),
                     "Score": f"{int(gf)}–{int(ga)}",
                     "Result": "W" if gf > ga else ("D" if gf == ga else "L")})
    return pd.DataFrame(rows)


def scoreline_heatmap(grid: np.ndarray, home: str, away: str, top: int = 6):
    t = theme.palette()
    sub = grid[:top, :top] * 100
    fig = px.imshow(sub, labels=dict(x=f"{away} goals", y=f"{home} goals", color="%"),
                    x=list(range(top)), y=list(range(top)), text_auto=".0f",
                    color_continuous_scale=t["ramp"], aspect="auto",
                    template=theme.register_template())
    fig.update_traces(hovertemplate=f"{home} %{{y}} – %{{x}} {away}<br>%{{z:.1f}}%<extra></extra>",
                      textfont=dict(size=11))
    fig.update_layout(height=300, coloraxis_showscale=False,
                      margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(showgrid=False, dtick=1)
    return fig


def fixture_detail(result, design, values, matches, fx, injuries):
    out_home, out_away = [], []
    if not values.empty:
        left, right = st.columns(2)
        for col, team, name, bucket in ((left, fx["home"], fx["home_name"], "home"),
                                        (right, fx["away"], fx["away_name"], "away")):
            squad = values[values["team"] == team].sort_values("att_share", ascending=False)
            flagged = availability.unavailable_for(injuries, values, team)
            options = squad["player"].tolist()[:30]
            picked = col.multiselect(f"{name} — unavailable", options,
                                     default=[p for p in flagged if p in options],
                                     key=f"out_{fx['home']}_{fx['away']}_{bucket}")
            (out_home if bucket == "home" else out_away).extend(picked)

    h_att, h_def = player_features.team_adjustment(values, fx["home"], out_home)
    a_att, a_def = player_features.team_adjustment(values, fx["away"], out_away)
    pred = simulate.predict_match(result, design, fx["home"], fx["away"], fx["league"],
                                  (h_att + a_def, a_att + h_def))
    if pred is None:
        st.warning("This fixture has no rating yet — newly promoted or an unmatched name.")
        return

    st.markdown(theme.outcome_bar_html(pred["home"], pred["draw"], pred["away"],
                                       fx["home_name"], fx["away_name"]),
                unsafe_allow_html=True)

    band = pred["home_hi"] - pred["home_lo"]
    if pd.notna(fx.get("odds_h")):
        book = 1 / np.array([fx["odds_h"], fx["odds_d"], fx["odds_a"]], dtype=float)
        book = book / book.sum()
        best = max(pred["home"] - book[0], pred["draw"] - book[1], pred["away"] - book[2])
        fourth = ("Edge vs book", f"{best*100:+.1f} pts",
                  f"book implies {book[0]*100:.0f}% home")
    elif band > 0.01:
        fourth = ("Home win range", f"{pred['home_lo']*100:.0f}–{pred['home_hi']*100:.0f}%",
                  "90% credible band")
    else:
        fourth = ("Draw chance", f"{pred['draw']*100:.0f}%", "point estimate, no odds")
    st.markdown(theme.tiles_html([
        ("Expected goals", f"{pred['exp_hg']:.2f} – {pred['exp_ag']:.2f}", "model rates"),
        ("Likeliest score", pred["top_score"], f"{pred['top_score_p']*100:.1f}% of outcomes"),
        ("Over 2.5", f"{pred['over_2.5']*100:.0f}%", f"both score {pred['btts']*100:.0f}%"),
        fourth,
    ]), unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        eyebrow("Scoreline probabilities")
        st.plotly_chart(scoreline_heatmap(pred["scoreline_grid"], fx["home_name"],
                                          fx["away_name"]), width="stretch")
    with right:
        eyebrow("Recent form")
        for team, name in ((fx["home"], fx["home_name"]), (fx["away"], fx["away_name"])):
            chips = theme.form_chips_html(form_results(matches, team))
            st.markdown(f"**{name}** &nbsp; {chips}", unsafe_allow_html=True)
            st.dataframe(recent_matches(matches, team), hide_index=True, width="stretch",
                         height=178)

    history = rivalry.head_to_head(matches, fx["home"], fx["away"]).head(6)
    if not history.empty:
        eyebrow("Head to head")
        table = pd.DataFrame({
            "Date": history["date"].dt.date,
            "Match": (history["home"].str.title() + "  " + history["hg"].astype(int).astype(str)
                      + "–" + history["ag"].astype(int).astype(str) + "  "
                      + history["away"].str.title()),
            "Season": history["season"],
        })
        st.dataframe(table, hide_index=True, width="stretch")


def page_fixtures(result, design, values, matches):
    fixtures = get_fixtures()
    if fixtures.empty:
        st.info("No upcoming fixtures in the feed right now.")
        return
    present = [c for c in COMPETITIONS if c in set(fixtures["league"])]
    chosen = league_filter("fx_leagues", present)
    fixtures = fixtures[fixtures["league"].isin(chosen)]
    if fixtures.empty:
        st.info("No fixtures for the selected leagues.")
        return

    preds = simulate.predict_fixtures(result, design, fixtures)
    if preds.empty:
        st.warning("No fixtures could be matched to rated teams.")
        return

    board = pd.DataFrame({
        "Kick-off": preds["date"].dt.strftime("%a %d %b"),
        "League": preds["league"].map(lambda c: COMPETITIONS.get(c, c)),
        "Match": preds["home_name"] + "  v  " + preds["away_name"],
        "Home": preds["p_home"] * 100, "Draw": preds["p_draw"] * 100,
        "Away": preds["p_away"] * 100,
        "Goals": preds["exp_hg"] + preds["exp_ag"],
    })
    eyebrow(f"{len(board)} upcoming fixtures — select one for detail")
    bar = st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                          width="small")
    picked = st.dataframe(
        board, hide_index=True, width="stretch", height=min(560, 60 + 35 * len(board)),
        on_select="rerun", selection_mode="single-row",
        column_config={"Home": bar, "Draw": bar, "Away": bar,
                       "Goals": st.column_config.NumberColumn(format="%.1f", width="small"),
                       "Match": st.column_config.TextColumn(width="medium")})

    rows = picked.selection.rows if picked and picked.selection else []
    if not rows:
        note("Pick a fixture above to see expected goals, scoreline probabilities, "
             "form and head-to-head.")
        return
    fx = preds.iloc[rows[0]]
    st.markdown(f"### {fx['home_name']} v {fx['away_name']}")
    st.markdown(f'<div class="fp-meta">{COMPETITIONS.get(fx["league"], fx["league"])}'
                f' · {fx["date"].strftime("%A %d %B %Y")}</div>', unsafe_allow_html=True)
    fixture_detail(result, design, values, matches, fx, get_injuries())


def page_ratings(result, design):
    table = hierarchical.ratings(result, design)
    table["league_name"] = table["league"].map(lambda c: COMPETITIONS.get(c, c))
    counts = table["league_name"].value_counts()
    ranked = [l for l in counts.index if counts[l] >= 8]
    league = st.pills("League", ranked, default=LEAGUE_NAMES["E0"], key="rating_league")
    sub = table[table["league_name"] == (league or LEAGUE_NAMES["E0"])].copy()
    sub["team"] = sub["team"].str.title()

    t = theme.palette()
    ranked = sub["strength"].rank(ascending=False)
    sub["label"] = np.where((ranked <= 5) | (ranked > len(sub) - 3), sub["team"], "")
    fig = px.scatter(sub, x="attack", y="defence", color="strength", text="label",
                     custom_data=["team"], color_continuous_scale=t["ramp"][1:], height=560,
                     template=theme.register_template(),
                     labels={"attack": "Attack  →  scores more",
                             "defence": "Defence  →  concedes less",
                             "strength": "Overall"})
    fig.update_traces(textposition="top center", marker=dict(size=11, line=dict(width=0)),
                      textfont=dict(size=11, color=t["ink_2"]),
                      hovertemplate="%{customdata[0]}<br>attack %{x:.2f} · "
                                    "defence %{y:.2f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dot", line_color=t["axis"], opacity=0.6)
    fig.add_vline(x=0, line_dash="dot", line_color=t["axis"], opacity=0.6)
    st.plotly_chart(fig, width="stretch")
    note("Zero is the league average; only the strongest and weakest sides are labelled — "
         "hover for the rest. Ratings are identified within a league, so a Bundesliga attack "
         "rating and a La Liga one are not on the same scale.")

    eyebrow("Full table")
    table_view = sub.drop(columns=["league_name", "attack_sd", "defence_sd", "label"]).round(3)
    st.dataframe(table_view.rename(columns={"team": "Team", "league": "League",
                                            "attack": "Attack", "defence": "Defence",
                                            "strength": "Overall"}),
                 hide_index=True, width="stretch")


def page_rivalry(result, design, matches):
    league = st.pills("League", list(LEAGUE_NAMES), default="E0",
                      format_func=lambda c: LEAGUE_NAMES[c], key="derby_league") or "E0"
    pairs = rivalry.DERBIES.get(league, [])
    if not pairs:
        st.info("No derbies listed for this league.")
        return
    labels = [f"{h.title()} v {a.title()} · {tag}" for h, a, tag in pairs]
    picked = st.selectbox("Fixture", range(len(labels)), format_func=lambda i: labels[i],
                          key="derby_pick")
    home, away, tag = pairs[picked]

    history = rivalry.head_to_head(matches, home, away)
    if history.empty:
        st.warning(f"No meetings between {home.title()} and {away.title()} since 2000.")
        return

    st.markdown(f"### {home.title()} v {away.title()}")
    st.markdown(f'<div class="fp-meta">{tag} · {len(history)} meetings since 2000</div>',
                unsafe_allow_html=True)

    pred = simulate.predict_match(result, design, home, away, league)
    if pred:
        st.markdown(theme.outcome_bar_html(pred["home"], pred["draw"], pred["away"],
                                           home.title(), away.title()),
                    unsafe_allow_html=True)

    split = rivalry.venue_split(matches, home, away)
    if not split.empty:
        eyebrow("Home and away split")
        split["fixture"] = split["fixture"].str.replace(
            r"^(\w[\w' ]*?) at home$", lambda m: f"{m.group(1).title()} at home", regex=True)
        shown = split.rename(columns={"fixture": "Fixture", "played": "Played",
                                      "host_wins": "Host W", "draws": "D",
                                      "guest_wins": "Guest W", "host_goals": "Host goals",
                                      "guest_goals": "Guest goals"})
        st.dataframe(shown, hide_index=True, width="stretch")

    venue_only = st.toggle(f"Only meetings at {home.title()}", value=True)
    grid = pred["scoreline_grid"] if pred else None
    table = rivalry.recurrence(matches, home, away, grid, venue_specific=venue_only)
    if table.empty:
        st.info("No meetings at this venue.")
        return

    eyebrow("Which scorelines actually recur")
    show = table.head(8).copy()
    show["share"] = show["share"] * 100
    if "model" in show:
        show["model"] = show["model"] * 100
    columns = {"score": "Score", "times": "Times", "share": "Historic %",
               "model": "Model %", "lift": "Lift"}
    st.dataframe(show[[c for c in columns if c in show]].rename(columns=columns),
                 hide_index=True, width="stretch",
                 column_config={"Historic %": st.column_config.NumberColumn(format="%.0f%%"),
                                "Model %": st.column_config.NumberColumn(format="%.1f%%"),
                                "Lift": st.column_config.NumberColumn(format="%.2f")})
    note("Lift above 1 means the scoreline has come up more often here than the model expects. "
         "Across all listed derbies, 52 scorelines clear a naive p&lt;0.05 test against about 26 "
         "expected by chance — and <b>none survive correction for multiple testing</b>. "
         "Read this as description, not prediction.")

    if grid is not None:
        eyebrow("Model scoreline probabilities")
        st.plotly_chart(scoreline_heatmap(grid, home.title(), away.title()), width="stretch")


def page_trajectory():
    try:
        params, design, _ = get_model("dynamic")
    except FileNotFoundError:
        st.info("Season-by-season ratings need the dynamic model.")
        st.code("python -m src.cli train --model dynamic")
        return
    from src.models import dynamic

    league = st.pills("League", list(LEAGUE_NAMES), default="E0",
                      format_func=lambda c: LEAGUE_NAMES[c], key="traj_league") or "E0"
    members = [t for i, t in enumerate(design.teams)
               if design.leagues[int(np.argmax(design.membership[i]))] == league]
    current = {t: params["att"][design.teams.index(t)] + params["def"][design.teams.index(t)]
               for t in members}
    members = sorted(members, key=lambda t: -current[t])
    chosen = st.multiselect("Teams", sorted(members), default=members[:4],
                            format_func=str.title, key="traj_teams")
    if not chosen:
        st.info("Pick at least one team.")
        return

    metric = st.segmented_control("Series", ["Overall", "Attack", "Defence"],
                                  default="Overall", key="traj_metric") or "Overall"
    frames = []
    for team in chosen:
        path = dynamic.trajectory(params, design, team)
        path["Team"] = team.title()
        frames.append(path)
    history = pd.concat(frames, ignore_index=True)
    history["Overall"] = history["attack"] + history["defence"]
    history = history.rename(columns={"attack": "Attack", "defence": "Defence",
                                      "season": "Season"})

    t = theme.palette()
    fig = px.line(history, x="Season", y=metric, color="Team", markers=True, height=520,
                  template=theme.register_template())
    fig.update_traces(line=dict(width=2), marker=dict(size=6))
    fig.add_hline(y=0, line_dash="dot", line_color=t["axis"], opacity=0.6)

    finals = [(team, history[history["Team"] == team].iloc[-1][metric])
              for team in history["Team"].unique()]
    spread = history[metric].max() - history[metric].min() or 1.0
    ordered = sorted(v for _, v in finals)
    crowded = any(b - a < spread * 0.06 for a, b in zip(ordered, ordered[1:]))
    if not crowded and len(finals) <= 4:
        for i, (team, value) in enumerate(finals):
            fig.add_annotation(x=history["Season"].iloc[-1], y=value, text=f"  {team}",
                               showarrow=False, xanchor="left",
                               font=dict(size=11, color=t["series"][i % len(t["series"])]))
    fig.update_layout(yaxis_title=metric, xaxis_title="",
                      showlegend=crowded or len(chosen) > 4, margin=dict(r=110))
    fig.update_xaxes(tickangle=0, dtick=4)
    st.plotly_chart(fig, width="stretch")
    note("Ratings follow an AR(1) path across seasons, so each line shows how a side's strength "
         "actually moved rather than one blended average. Zero is the league average that season.")


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
    log_loss = -np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-12, 1)).mean()

    mask = preds[["odds_h", "odds_d", "odds_a"]].notna().all(axis=1)
    book_loss = None
    if mask.sum() > 50:
        book = implied_probabilities(preds[mask])
        book_loss = -np.log(book[np.arange(int(mask.sum())), actual[mask.to_numpy()]]).mean()

    st.markdown(theme.tiles_html([
        ("Matches tested", f"{len(preds):,}", "walk-forward, out of sample"),
        ("Log loss", f"{log_loss:.4f}", "lower is better · uniform = 1.0986"),
        ("Bookmaker", f"{book_loss:.4f}" if book_loss else "—", "the bar to beat"),
        ("Accuracy", f"{(probs.argmax(1) == actual).mean():.1%}",
         f"always-home = {(actual == 0).mean():.1%}"),
    ]), unsafe_allow_html=True)
    note("The model has real skill but does not beat the market. Betting its disagreements with "
         "the book returns −8% to −14% against a 5.2% overround, and gets worse as the claimed "
         "edge grows. <b>Treat this as an analysis tool, not a staking system.</b>")

    frame = pd.DataFrame({"p": probs.ravel(),
                          "hit": np.eye(3)[actual].ravel()})
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, 11))
    cal = frame.groupby("bin", observed=True).agg(predicted=("p", "mean"),
                                                  observed=("hit", "mean"),
                                                  n=("hit", "size")).dropna()
    t = theme.palette()
    eyebrow("Calibration — predicted probability vs what actually happened")
    fig = px.scatter(cal, x="predicted", y="observed", size="n", height=460,
                     template=theme.register_template(),
                     labels={"predicted": "Predicted probability",
                             "observed": "Observed frequency"})
    fig.update_traces(marker=dict(color=t["home"], line=dict(width=0)),
                      hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<extra></extra>")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dot", color=t["axis"]))
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    note("Points on the dotted line mean the stated probabilities are honest: outcomes given a "
         "30% chance happen about 30% of the time. Bubble size is the number of predictions.")


def main():
    theme.inject_css()
    st.markdown("# Elite Scores")
    try:
        result, design, variant = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m src.cli build\npython -m src.cli train --fast")
        return

    matches = get_matches()
    st.markdown(
        f'<div class="fp-sub">{len(matches):,} matches · {matches["date"].min().year}–'
        f'{matches["date"].max().year} · {len(LEAGUE_NAMES)} leagues · {variant} model</div>',
        unsafe_allow_html=True)

    page = st.sidebar.radio("View", ["Fixtures", "Team ratings", "Derbies & H2H",
                                     "Trajectory", "Backtest"], label_visibility="collapsed")
    if page == "Fixtures":
        page_fixtures(result, design, get_values(), matches)
    elif page == "Team ratings":
        page_ratings(result, design)
    elif page == "Derbies & H2H":
        page_rivalry(result, design, matches)
    elif page == "Trajectory":
        page_trajectory()
    else:
        page_backtest()


if __name__ == "__main__":
    main()
