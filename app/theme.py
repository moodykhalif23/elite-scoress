from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

LIGHT = {
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink_2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)", "wash": "rgba(11,11,11,0.04)",
    "home": "#2a78d6", "draw": "#898781", "away": "#e34948",
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#4a3aa7", "#e34948"],
    "ramp": ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"],
    "good": "#0ca30c", "critical": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink_2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "border": "rgba(255,255,255,0.10)", "wash": "rgba(255,255,255,0.05)",
    "home": "#3987e5", "draw": "#898781", "away": "#e66767",
    "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
               "#9085e9", "#e66767"],
    "ramp": ["#1a1a19", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
    "good": "#0ca30c", "critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette() -> dict:
    mode = getattr(getattr(st.context, "theme", None), "type", "light")
    return DARK if mode == "dark" else LIGHT


def register_template() -> str:
    for name, tokens in (("fp_light", LIGHT), ("fp_dark", DARK)):
        pio.templates[name] = go.layout.Template(layout=go.Layout(
            font=dict(family=FONT, size=13, color=tokens["ink_2"]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            colorway=tokens["series"],
            margin=dict(l=8, r=8, t=28, b=8),
            xaxis=dict(gridcolor=tokens["grid"], zerolinecolor=tokens["axis"],
                       linecolor=tokens["axis"], tickfont=dict(color=tokens["muted"]),
                       title_font=dict(color=tokens["ink_2"], size=12)),
            yaxis=dict(gridcolor=tokens["grid"], zerolinecolor=tokens["axis"],
                       linecolor=tokens["axis"], tickfont=dict(color=tokens["muted"]),
                       title_font=dict(color=tokens["ink_2"], size=12)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(color=tokens["ink_2"])),
            hoverlabel=dict(font_family=FONT, font_size=12),
        ))
    mode = getattr(getattr(st.context, "theme", None), "type", "light")
    return "fp_dark" if mode == "dark" else "fp_light"


def inject_css() -> None:
    t = palette()
    st.markdown(f"""<style>
      :root {{
        --fp-surface: {t['surface']}; --fp-ink: {t['ink']}; --fp-ink-2: {t['ink_2']};
        --fp-muted: {t['muted']}; --fp-border: {t['border']}; --fp-wash: {t['wash']};
        --fp-home: {t['home']}; --fp-draw: {t['draw']}; --fp-away: {t['away']};
      }}
      .block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1180px; }}
      #MainMenu, footer {{ visibility: hidden; }}
      h1, h2, h3, h4 {{ font-family: {FONT}; letter-spacing: -0.015em; }}
      h1 {{ font-size: 1.8rem !important; font-weight: 650 !important; margin-bottom: .1rem; }}
      h3 {{ font-size: 1.05rem !important; font-weight: 600 !important;
            margin: 1.6rem 0 .5rem; }}
      h4 {{ font-size: .95rem !important; font-weight: 600 !important; }}

      .fp-sub {{ color: var(--fp-muted); font-size: .88rem; margin-bottom: 1.6rem; }}
      .fp-eyebrow {{ color: var(--fp-muted); font-size: .72rem; font-weight: 600;
                     letter-spacing: .09em; text-transform: uppercase;
                     margin: 1.8rem 0 .55rem; }}

      .fp-card {{ border: 1px solid var(--fp-border); border-radius: 12px;
                  padding: 1rem 1.15rem; background: var(--fp-surface); }}

      .fp-teams {{ display: flex; align-items: baseline; gap: .5rem;
                   font-size: 1.15rem; font-weight: 600; color: var(--fp-ink); }}
      .fp-vs {{ color: var(--fp-muted); font-weight: 400; font-size: .9rem; }}
      .fp-meta {{ color: var(--fp-muted); font-size: .8rem; margin-top: .15rem; }}

      .fp-bar {{ display: flex; gap: 2px; height: 34px; margin: .85rem 0 .3rem; }}
      .fp-seg {{ display: flex; align-items: center; justify-content: center;
                 font-size: .78rem; font-weight: 600; color: #fff; overflow: hidden;
                 white-space: nowrap; }}
      .fp-seg:first-child {{ border-radius: 5px 2px 2px 5px; }}
      .fp-seg:last-child {{ border-radius: 2px 5px 5px 2px; }}
      .fp-legend {{ display: flex; gap: 1.1rem; color: var(--fp-muted);
                    font-size: .74rem; margin-bottom: .2rem; }}
      .fp-key {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px;
                 margin-right: .35rem; }}

      .fp-tiles {{ display: flex; gap: .6rem; flex-wrap: wrap; margin: .4rem 0 1rem; }}
      .fp-tile {{ flex: 1 1 150px; border: 1px solid var(--fp-border); border-radius: 10px;
                  padding: .75rem .9rem; background: var(--fp-surface); }}
      .fp-tile-label {{ color: var(--fp-muted); font-size: .72rem; font-weight: 600;
                        letter-spacing: .05em; text-transform: uppercase; }}
      .fp-tile-value {{ color: var(--fp-ink); font-size: 1.45rem; font-weight: 640;
                        line-height: 1.25; margin-top: .2rem; }}
      .fp-tile-note {{ color: var(--fp-muted); font-size: .76rem; }}

      .fp-form {{ display: inline-flex; gap: 3px; }}
      .fp-chip {{ width: 20px; height: 20px; border-radius: 4px; color: #fff;
                  font-size: .7rem; font-weight: 700; display: inline-flex;
                  align-items: center; justify-content: center; }}
      .fp-note {{ color: var(--fp-muted); font-size: .8rem; line-height: 1.5; }}

      div[data-testid="stDataFrame"] {{ border-radius: 10px; }}
      section[data-testid="stSidebar"] {{ border-right: 1px solid var(--fp-border); }}
      section[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}
    </style>""", unsafe_allow_html=True)


def outcome_bar_html(home: float, draw: float, away: float,
                     home_name: str, away_name: str) -> str:
    t = palette()
    parts = [(home, t["home"], "Home"), (draw, t["draw"], "Draw"), (away, t["away"], "Away")]
    segments = "".join(
        f'<div class="fp-seg" style="width:{share*100:.4f}%;background:{colour}">'
        f'{share*100:.0f}%</div>' for share, colour, _ in parts if share > 0)
    legend = (
        f'<div class="fp-legend">'
        f'<span><span class="fp-key" style="background:{t["home"]}"></span>{home_name}</span>'
        f'<span><span class="fp-key" style="background:{t["draw"]}"></span>Draw</span>'
        f'<span><span class="fp-key" style="background:{t["away"]}"></span>{away_name}</span>'
        f'</div>')
    return f'<div class="fp-bar">{segments}</div>{legend}'


def tiles_html(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="fp-tile"><div class="fp-tile-label">{label}</div>'
        f'<div class="fp-tile-value">{value}</div>'
        f'<div class="fp-tile-note">{note}</div></div>' for label, value, note in items)
    return f'<div class="fp-tiles">{cells}</div>'


def form_chips_html(results: list[str]) -> str:
    t = palette()
    colours = {"W": t["good"], "D": t["muted"], "L": t["critical"]}
    chips = "".join(f'<span class="fp-chip" style="background:{colours.get(r, t["muted"])}">'
                    f'{r}</span>' for r in results)
    return f'<span class="fp-form">{chips}</span>'
