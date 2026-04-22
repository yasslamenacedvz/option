"""
GEXRADAR // QUANT TERMINAL  — Premium Redesign
Run: streamlit run app.py

─────────────────────────────────────────────────────────────────────────────
GEX FORMULA (Perfiliev / SpotGamma / Barchart industry standard):
  GEX = Gamma × OI × ContractSize(100) × Spot² × 0.01 / 1e9

  This gives dealer dollar-exposure per 1% spot move, in billions.
  Calls: +GEX  (dealers assumed long calls → long gamma → stabilising)
  Puts:  -GEX  (dealers assumed short puts → short gamma → destabilising)

FIXES APPLIED vs prior version:
  1. Strike range aligned to ±8% in BOTH _process_chain AND bar_layout.
     (Old ±12% in calc + ±8% chart = phantom OOB contributions to cum-sum.)
  2. OI minimum raised 10 → 100. At OI=10 near-expiry ATM gamma is enormous;
     tiny positions made prominent bars. 100 is consistent with SpotGamma.
  3. IV gating: require bid>0 AND ask>0 AND mid>0.05 before solving IV.
     Zero-bid OTM options previously got a full smile fallback IV applied,
     producing phantom GEX bars with no real market behind them.
  4. Fallback IV now ONLY applied when mid>0.05 (real market exists).
     No valid IV + no real market → skip the strike entirely.
  5. IV cap tightened 2.5→1.5. 250% IV on equity options is unrealistic
     and could still inflate gamma on edge-case strikes.
  6. Heatmap fetch_options_data_heatmap: added 90 DTE cap (was missing).
  7. ES/NQ equiv conversion: now uses DYNAMIC live ratio (ES_price/SPY_price,
     NQ_price/QQQ_price) fetched from Yahoo Finance, NOT a hardcoded multiplier.
     Ratio floats daily with dividends, carry, and index drift. Falls back
     to last known ratio if fetch fails.
  8. Auto-refresh: page reruns every 60 seconds to keep GEX levels current.
  9. Cache TTL reduced from 900s to 60s to prevent stale data on cloud deployments.
─────────────────────────────────────────────────────────────────────────────
"""

import math
import time
import datetime
import warnings
import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.043  # ~Fed funds rate as of early 2025; update as rates change
DIV_YIELD = {
    "SPY": 0.013, "QQQ": 0.006, "IWM": 0.012,
    "GLD": 0.0,   "SLV": 0.0,   "TLT": 0.04,
    "XLF": 0.018, "XLE": 0.035, "IBIT": 0.0,
    "AAPL": 0.005,"NVDA": 0.001,"TSLA": 0.0,
    "AMZN": 0.0,  "MSFT": 0.007,"META": 0.004,
    "GOOGL": 0.0, "SPX": 0.013, "NDX": 0.006,
    "RUT": 0.012,
}

# Auto-refresh interval in seconds
AUTO_REFRESH_SECONDS = 60

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GEXRADAR", layout="wide", initial_sidebar_state="expanded")

if "current_page" not in st.session_state:
    st.session_state.current_page = "DASHBOARD"
if "radar_mode" not in st.session_state:
    st.session_state.radar_mode = "GEX"
if "asset_choice" not in st.session_state:
    st.session_state.asset_choice = "SPY"
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

st.session_state.current_page = "DASHBOARD"
if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Default"


# ─────────────────────────────────────────────────────────────────────────────
# THEME SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
THEMES = {
    # ── Original dark navy/teal ─────────────────────────────────────────────
    "Default": dict(
        bg="#03070D", bg1="#070E17", bg2="#0B1420", bg3="#0F1C2E",
        line="#111E2E", line2="#192D44", line_bright="#1F3A58",
        t1="#C8DCEF",  t2="#7FA8C4",  t3="#4E7A9C",  t4="#2A4260",
        green="#00E5A0", red="#FF3860", amber="#F5A623", blue="#4A9FFF", violet="#9B7FFF",
        green_glow="rgba(0,229,160,0.10)", red_glow="rgba(255,56,96,0.10)",
        nav_bg="rgba(3,7,13,0.97)", nav_border="rgba(255,255,255,0.05)",
        nav_t1="#C8DCEF", nav_t2="#3D6680", nav_clock="#5A8AA8", nav_dot="#00E5A0",
        chart_bg="#03070D", chart_line="#111E2E", chart_line2="#192D44",
        chart_t2="#7FA8C4", chart_t3="#4E7A9C",
        chart_hover="#0F1C2E", chart_hover_border="#192D44", chart_hover_text="#C8DCEF",
        bar_pos="#00E5A0", bar_neg="#FF3860", spot_line="#C8DCEF", gamma_line="#4E7A9C",
        surface_colorscale=[
            [0.00,"#FF0040"],[0.15,"#CC0030"],[0.30,"#7A0020"],
            [0.45,"#1A0308"],[0.50,"#03070D"],[0.55,"#001A0A"],
            [0.70,"#006633"],[0.85,"#00CC77"],[1.00,"#00FF99"],
        ],
        heat_colorscale=[
            [0.00,"#FF0040"],[0.20,"#CC0030"],[0.38,"#550015"],
            [0.48,"#150005"],[0.50,"#03070D"],[0.52,"#001505"],
            [0.62,"#005530"],[0.80,"#00CC70"],[1.00,"#00FF99"],
        ],
    ),
    # ── Pure black with green/red signals ──────────────────────────────────
    "Obsidian": dict(
        bg="#000000", bg1="#080808", bg2="#101010", bg3="#181818",
        line="rgba(255,255,255,0.06)", line2="rgba(255,255,255,0.11)", line_bright="rgba(255,255,255,0.22)",
        t1="#F0F0F0",  t2="#888888",  t3="#4A4A4A",  t4="#2A2A2A",
        green="#00FF88", red="#FF3355", amber="#FFB800", blue="#5599FF", violet="#AA88FF",
        green_glow="rgba(0,255,136,0.08)", red_glow="rgba(255,51,85,0.08)",
        nav_bg="#000000", nav_border="rgba(255,255,255,0.07)",
        nav_t1="#F0F0F0", nav_t2="#4A4A4A", nav_clock="#777777", nav_dot="#00FF88",
        chart_bg="#000000", chart_line="rgba(255,255,255,0.06)", chart_line2="rgba(255,255,255,0.11)",
        chart_t2="#888888", chart_t3="#444444",
        chart_hover="#111111", chart_hover_border="rgba(255,255,255,0.12)", chart_hover_text="#F0F0F0",
        bar_pos="#00FF88", bar_neg="#FF3355", spot_line="#F0F0F0", gamma_line="#333333",
        surface_colorscale=[
            [0.00,"#FF3355"],[0.20,"#CC1133"],[0.40,"#440011"],
            [0.50,"#000000"],[0.60,"#004422"],[0.80,"#00CC66"],[1.00,"#00FF88"],
        ],
        heat_colorscale=[
            [0.00,"#FF3355"],[0.25,"#881122"],[0.45,"#220008"],
            [0.50,"#000000"],[0.55,"#002211"],[0.75,"#008844"],[1.00,"#00FF88"],
        ],
    ),
    # ── Clean white — white/gray backgrounds, real signal colors ─────────────
    "Light": dict(
        bg="#FFFFFF", bg1="#F5F5F5", bg2="#EBEBEB", bg3="#E0E0E0",
        line="rgba(0,0,0,0.08)", line2="rgba(0,0,0,0.15)", line_bright="rgba(0,0,0,0.32)",
        t1="#0A0A0A",  t2="#444444",  t3="#888888",  t4="#BBBBBB",
        green="#00A86B", red="#D93025", amber="#B8860B", blue="#1A56CC", violet="#7B42A8",
        green_glow="rgba(0,168,107,0.12)", red_glow="rgba(217,48,37,0.12)",
        nav_bg="#FFFFFF", nav_border="rgba(0,0,0,0.09)",
        nav_t1="#0A0A0A", nav_t2="#888888", nav_clock="#555555", nav_dot="#00A86B",
        chart_bg="#FFFFFF", chart_line="rgba(0,0,0,0.08)", chart_line2="rgba(0,0,0,0.15)",
        chart_t2="#555555", chart_t3="#999999",
        chart_hover="#F0F0F0", chart_hover_border="rgba(0,0,0,0.18)", chart_hover_text="#0A0A0A",
        bar_pos="#00A86B", bar_neg="#D93025", spot_line="#0A0A0A", gamma_line="#AAAAAA",
        surface_colorscale=[
            [0.00,"#D93025"],[0.25,"#E8776F"],[0.48,"#F5D5D3"],
            [0.50,"#FFFFFF"],[0.52,"#D4EDE3"],[0.75,"#66C2A5"],[1.00,"#00A86B"],
        ],
        heat_colorscale=[
            [0.00,"#D93025"],[0.25,"#E8776F"],[0.47,"#F5E5E4"],
            [0.50,"#FFFFFF"],[0.53,"#E0F0E8"],[0.75,"#55BB8A"],[1.00,"#00A86B"],
        ],
    ),
    # ── Warm amber phosphor terminal ────────────────────────────────────────
    "Amber": dict(
        bg="#060400", bg1="#0E0A00", bg2="#161000", bg3="#1E1600",
        line="rgba(255,180,0,0.09)", line2="rgba(255,180,0,0.17)", line_bright="rgba(255,180,0,0.35)",
        t1="#FFE066",  t2="#CC9900",  t3="#7A5A00",  t4="#3D2D00",
        green="#FFD700", red="#FF6600", amber="#FFB800", blue="#FFCC44", violet="#FF9933",
        green_glow="rgba(255,215,0,0.08)", red_glow="rgba(255,102,0,0.08)",
        nav_bg="#060400", nav_border="rgba(255,180,0,0.10)",
        nav_t1="#FFE066", nav_t2="#7A5A00", nav_clock="#CC9900", nav_dot="#FFB800",
        chart_bg="#060400", chart_line="rgba(255,180,0,0.09)", chart_line2="rgba(255,180,0,0.17)",
        chart_t2="#CC9900", chart_t3="#664400",
        chart_hover="#161000", chart_hover_border="rgba(255,180,0,0.2)", chart_hover_text="#FFE066",
        bar_pos="#FFD700", bar_neg="#FF6600", spot_line="#FFE066", gamma_line="#664400",
        surface_colorscale=[
            [0.00,"#FF6600"],[0.30,"#AA3300"],[0.48,"#221100"],
            [0.50,"#060400"],[0.52,"#221400"],[0.70,"#AA7700"],[1.00,"#FFD700"],
        ],
        heat_colorscale=[
            [0.00,"#FF6600"],[0.25,"#882200"],[0.47,"#1A0A00"],
            [0.50,"#060400"],[0.53,"#1A0E00"],[0.75,"#997700"],[1.00,"#FFD700"],
        ],
    ),
    # ── Deep arctic blue ────────────────────────────────────────────────────
    "Glacier": dict(
        bg="#030710", bg1="#060D1C", bg2="#0A1428", bg3="#0E1B34",
        line="rgba(100,160,230,0.09)", line2="rgba(100,160,230,0.16)", line_bright="rgba(100,160,230,0.32)",
        t1="#D8ECFF",  t2="#6A9EC8",  t3="#3A6090",  t4="#1A3660",
        green="#44CCFF", red="#FF4466", amber="#66AAFF", blue="#44CCFF", violet="#8866FF",
        green_glow="rgba(68,204,255,0.09)", red_glow="rgba(255,68,102,0.09)",
        nav_bg="#030710", nav_border="rgba(100,160,230,0.09)",
        nav_t1="#D8ECFF", nav_t2="#3A6090", nav_clock="#6A9EC8", nav_dot="#44CCFF",
        chart_bg="#030710", chart_line="rgba(100,160,230,0.09)", chart_line2="rgba(100,160,230,0.16)",
        chart_t2="#6A9EC8", chart_t3="#3A6090",
        chart_hover="#0A1428", chart_hover_border="rgba(100,160,230,0.2)", chart_hover_text="#D8ECFF",
        bar_pos="#44CCFF", bar_neg="#FF4466", spot_line="#D8ECFF", gamma_line="#3A6090",
        surface_colorscale=[
            [0.00,"#FF4466"],[0.20,"#CC1133"],[0.40,"#330011"],
            [0.50,"#030710"],[0.60,"#001A44"],[0.80,"#0088CC"],[1.00,"#44CCFF"],
        ],
        heat_colorscale=[
            [0.00,"#FF4466"],[0.25,"#881122"],[0.45,"#150018"],
            [0.50,"#030710"],[0.55,"#001022"],[0.75,"#005599"],[1.00,"#44CCFF"],
        ],
    ),
}
def get_theme():
    name = st.session_state.get("ui_theme", "Default")
    return THEMES.get(name, THEMES["Default"])


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
# ── Dynamic theme CSS injection ────────────────────────────────────────────
_T = get_theme()
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&family=Barlow:wght@300;400;500;600&display=swap');
:root {{
    --bg:            {_T['bg']};
    --base:          {_T['bg1']};
    --surface:       {_T['bg2']};
    --surface-2:     {_T['bg2']};
    --surface-3:     {_T['bg3']};
    --void:          {_T['bg']};
    --line:          {_T['line']};
    --line-2:        {_T['line2']};
    --line-bright:   {_T['line_bright']};
    --text-1:        {_T['t1']};
    --text-2:        {_T['t2']};
    --text-3:        {_T['t3']};
    --text-4:        {_T['t4']};
    --green:         {_T['green']};
    --green-dim:     {_T['green']};
    --green-glow:    {_T['green_glow']};
    --red:           {_T['red']};
    --red-dim:       {_T['red']};
    --red-glow:      {_T['red_glow']};
    --amber:         {_T['amber']};
    --blue:          {_T['blue']};
    --violet:        {_T['violet']};
    --mono:          'JetBrains Mono', monospace;
    --display:       'Barlow Condensed', sans-serif;
    --body:          'Barlow', sans-serif;
    --radius:        2px;
    --radius-lg:     3px;
    --radius-xl:     4px;
    --transition:    all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  /* aliases for legacy class refs */
  --bg-1: {_T['bg1']};
  --bg-2: {_T['bg2']};
  --bg-3: {_T['bg3']};
}}
</style>
""", unsafe_allow_html=True)

# ── Update module-level Plotly vars ────────────────────────────────────────
_TC  = get_theme()
BG   = _TC["chart_bg"]
LINE = _TC["chart_line"]
LINE2= _TC["chart_line2"]
TEXT2= _TC["chart_t2"]
TEXT3= _TC["chart_t3"]


st.markdown("""
<style>
*, *::before, *::after { box-sizing: border-box; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
html, body, [class*="css"] {
    font-family: var(--body);
    background-color: var(--bg);
    color: var(--text-1);
    -webkit-font-smoothing: antialiased;
}

section[data-testid="stSidebar"] {
    background: var(--base) !important;
    border-right: 1px solid var(--line) !important;
    min-width: 240px !important;
    max-width: 240px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Show sidebar navigation"],
[data-testid="stSidebarNavCollapseButton"] {
    display: none !important;
    pointer-events: none !important;
}

.exp-dial-wrap { padding: 10px 0 6px 0; }
.exp-dial-header { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; }
.exp-dial-label { font-family:var(--body); font-size:9px; color:var(--text-3); letter-spacing:1.5px; text-transform:uppercase; }
.exp-dial-value { font-family:var(--mono); font-size:18px; font-weight:500; color:var(--text-1); letter-spacing:-1px; line-height:1; }
.exp-pip-row { display:flex; gap:4px; align-items:flex-end; height:28px; }
.exp-pip { flex:1; border-radius:1px; transition:var(--transition); cursor:pointer; }
.exp-pip.active { background:var(--text-1); }
.exp-pip.inactive { background:var(--bg-3); border:1px solid var(--line); }
.exp-pip.inactive:hover { background:var(--bg-2); }
.dual-btn-wrap button {
    font-family: var(--mono) !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid var(--line2) !important;
    color: var(--text-3) !important;
    border-radius: 3px !important;
    padding: 6px 12px !important;
    height: auto !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
    cursor: pointer !important;
    user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important;
    margin-bottom: 12px !important;
}
.dual-btn-wrap button:hover {
    border-color: var(--text-3) !important;
    color: var(--text-1) !important;
    background: var(--bg-2) !important;
}
.dual-btn-wrap button:active {
    transform: scale(.97) !important;
    transition: transform .04s ease !important;
}
.dual-btn-wrap button:focus,
.dual-btn-wrap button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}

.sb-section {
    font-family: var(--body);
    font-size: 8px;
    font-weight: 600;
    color: var(--text-4);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 20px 0 7px 0;
    margin: 0;
    border-bottom: 1px solid var(--line);
}
.sb-section:first-child { padding-top: 14px; }

.m-tile {
    padding: 7px 0;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    transition: var(--transition);
}
.m-tile:last-child { border-bottom: none; }
.m-label {
    font-family: var(--body);
    font-size: 9px;
    color: var(--text-3);
    letter-spacing: 0.4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 1;
    min-width: 0;
}
.m-value {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: -0.3px;
    white-space: nowrap;
    flex-shrink: 0;
}
.sb-group { padding: 4px 0 10px 0; }

.regime {
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    overflow: hidden;
}
.regime::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    height: 1px;
    width: 100%;
    background: var(--bar-grad);
    opacity: 0.5;
}
.regime-meta { font-family: var(--mono); font-size: 9px; color: var(--text-3); letter-spacing: 1.2px; margin-bottom: 6px; text-transform: uppercase; }
.regime-state { font-family: var(--display); font-size: 22px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
.regime-right { text-align: right; }
.regime-bias-label { font-family: var(--body); font-size: 8.5px; color: var(--text-3); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 5px; }
.regime-bias-value { font-family: var(--display); font-size: 15px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }

.mode-btn-row { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
.mode-btn {
    font-family: var(--mono); font-size: 9px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;
    padding: 7px 14px; border-radius: var(--radius); border: 1px solid var(--line);
    background: transparent; color: var(--text-3); cursor: pointer; transition: var(--transition);
    white-space: nowrap;
}
.mode-btn:hover { border-color: var(--line-bright); color: var(--text-1); background: var(--bg-2); }
.mode-btn.active { border-color: var(--text-1); color: var(--text-1); background: var(--bg-2); }

div.stButton > button {
    background: transparent !important;
    border: 1px solid var(--line2) !important;
    color: var(--text-3) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: .8px !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    height: 34px !important;
    width: 100% !important;
    cursor: pointer !important;
    user-select: none !important;
    white-space: nowrap !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important;
}
div.stButton > button:hover {
    background: var(--bg-2) !important;
    border-color: var(--text-3) !important;
    color: var(--text-1) !important;
}
div.stButton > button:active {
    transform: scale(.97) !important;
    transition: transform .04s ease !important;
}
div.stButton > button:focus,
div.stButton > button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
}
div.stButton > button[kind="primary"] {
    background: var(--bg-2) !important;
    border-color: var(--line-bright) !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
}
div.stButton > button[kind="primary"]:hover {
    border-color: var(--text-1) !important;
}
div.stButton > button[kind="primary"]:active {
    transform: scale(.97) !important;
}
/* kill the Streamlit spinner/loading overlay that flashes on button click */
div.stButton > button > div[data-testid="stSpinner"],
div.stButton > button svg,
div.stButton > button .stMarkdown { pointer-events: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

.kl-panel { background: var(--bg-1); border: 1px solid var(--line); border-radius: var(--radius-lg); overflow: hidden; }
.kl-header { padding: 11px 16px; border-bottom: 1px solid var(--line); font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3); letter-spacing: 2.5px; text-transform: uppercase; background: var(--bg); display: flex; align-items: center; gap: 8px; }
.kl-header::before { content: ''; width: 14px; height: 1px; background: var(--text-1); flex-shrink: 0; }
.kl-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); transition: var(--transition); cursor: default; }
.kl-row:hover { background: var(--bg-2); }
.kl-row:last-child { border-bottom: none; }
.kl-name { font-family: var(--body); font-size: 9.5px; color: var(--text-2); letter-spacing: 0.3px; }
.kl-val { font-family: var(--mono); font-size: 11px; font-weight: 500; letter-spacing: -0.3px; }

.sec-head {
    font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3);
    letter-spacing: 3px; text-transform: uppercase; padding: 22px 0 11px 0;
    border-bottom: 1px solid var(--line); margin-bottom: 16px; position: relative;
    display: flex; align-items: center; gap: 10px;
}
.sec-head::before { content: ''; width: 16px; height: 1px; background: var(--text-1); flex-shrink: 0; }

.sub-head {
    font-family: var(--mono); font-size: 8px; font-weight: 600; color: var(--text-3);
    letter-spacing: 2.5px; text-transform: uppercase; padding: 12px 0 9px 0;
    border-bottom: 1px solid var(--line); margin-bottom: 12px;
}

.stDataFrame { border: none !important; }
.stDataFrame thead tr th { font-family: var(--mono) !important; font-size: 8.5px !important; font-weight: 600 !important; color: var(--text-3) !important; letter-spacing: 1.8px !important; text-transform: uppercase !important; background: var(--bg) !important; border-bottom: 1px solid var(--line) !important; padding: 9px 11px !important; }
.stDataFrame tbody tr td { font-family: var(--mono) !important; font-size: 10.5px !important; padding: 8px 11px !important; border-bottom: 1px solid var(--line) !important; background: var(--base) !important; color: var(--text-1) !important; }
.stDataFrame tbody tr:hover td { background: var(--bg-2) !important; }

div[data-baseweb="select"] > div { background: var(--base) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important; font-family: var(--mono) !important; font-size: 11px !important; transition: var(--transition) !important; }
div[data-baseweb="select"] > div:hover { border-color: var(--line-bright) !important; }
div[data-baseweb="select"] span { color: var(--text-2) !important; font-family: var(--mono) !important; font-size: 11px !important; }
div[data-testid="stRadio"] label { font-family: var(--mono) !important; font-size: 11px !important; color: var(--text-2) !important; }

div[data-testid="stDownloadButton"] button {
    background: transparent !important; border: 1px solid var(--line2) !important;
    color: var(--text-3) !important; font-family: var(--mono) !important;
    font-size: 10px !important; font-weight: 600 !important; letter-spacing: .8px !important;
    text-transform: uppercase !important; border-radius: 3px !important;
    cursor: pointer !important; user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important; }
div[data-testid="stDownloadButton"] button:hover { border-color: var(--text-3) !important; color: var(--text-1) !important; background: var(--bg-2) !important; }
div[data-testid="stDownloadButton"] button:active { transform: scale(.97) !important; transition: transform .04s ease !important; }
div[data-testid="stDownloadButton"] button:focus, div[data-testid="stDownloadButton"] button:focus-visible { outline: none !important; box-shadow: none !important; }

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 1px; }
::-webkit-scrollbar-thumb:hover { background: var(--line-bright); }

div[data-testid="stSpinner"] { color: var(--text-3) !important; }
div[data-testid="stSelectbox"] label { display: none !important; }

section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button {
    height: 30px !important; font-size: 11px !important; font-family: var(--mono) !important;
    font-weight: 600 !important; letter-spacing: 1.5px !important; border-radius: 20px !important;
    padding: 0 !important; user-select: none !important;
    transition: border-color .07s ease, color .07s ease, background .07s ease !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:active { transform: scale(.96) !important; transition: transform .04s ease !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:focus,
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button:focus-visible { outline: none !important; box-shadow: none !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="primary"] { background: var(--bg-3) !important; border-color: var(--line-bright) !important; color: var(--text-1) !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"] { background: transparent !important; border-color: var(--line) !important; color: var(--text-3) !important; }
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] div.stButton > button[kind="secondary"]:hover { border-color: var(--line-bright) !important; color: var(--text-2) !important; }

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] { margin-top: 8px; }
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label { display: none !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div { border-radius: 20px !important; padding: 0 12px !important; min-height: 32px !important; height: 32px !important; font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1.5px !important; background: var(--base) !important; border-color: var(--line-bright) !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus,
section[data-testid="stSidebar"] div[data-baseweb="select"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"] [data-testid="stWidgetLabel"] { outline: none !important; box-shadow: none !important; caret-color: transparent !important; animation: none !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] span { color: var(--text-1) !important; font-family: var(--mono) !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1.5px !important; }
div[data-baseweb="popover"] ul[role="listbox"] li { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 1px !important; color: var(--text-2) !important; background: var(--base) !important; }
div[data-baseweb="popover"] ul[role="listbox"] li:hover { background: var(--bg-2) !important; color: var(--text-1) !important; }
div[data-baseweb="popover"] ul[role="listbox"] li[data-value^="──"] { font-family: var(--mono) !important; font-size: 9px !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: var(--text-3) !important; background: #050505 !important; padding-top: 10px !important; padding-bottom: 4px !important; pointer-events: none !important; cursor: default !important; border-top: 1px solid var(--line) !important; }

.kl-row-full { display: flex; align-items: center; justify-content: space-between; padding: 9px 16px; border-bottom: 1px solid var(--line); transition: var(--transition); }
.kl-row-full:hover { background: var(--bg-2); }
.kl-row-full:last-child { border-bottom: none; }
.kl-row-full-left { display: flex; align-items: center; gap: 10px; }
.kl-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }
.kl-row-full .kl-name { font-family: var(--body); font-size: 10px; font-weight: 500; letter-spacing: 0.3px; }
.kl-price-val { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: -0.3px; }

.dual-toggle-row { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; }
.dual-pill { display: inline-flex; align-items: center; gap: 8px; background: transparent; border: 1px solid var(--line); border-radius: 24px; padding: 6px 14px 6px 10px; font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: 1.2px; color: var(--text-3); cursor: pointer; transition: var(--transition); text-transform: uppercase; user-select: none; }
.dual-pill:hover { border-color: var(--line-bright); color: var(--text-2); }
.dual-pill.active { border-color: var(--text-1); color: var(--text-1); background: var(--bg-2); }
.dual-pip { width: 6px; height: 6px; border-radius: 50%; background: var(--text-3); transition: var(--transition); }
.dual-pill.active .dual-pip { background: var(--text-1); }
.dual-pip-pair { display: flex; gap: 3px; }

/* ── Theme select pill — slightly larger & distinct from asset picker ── */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"]:first-of-type div[data-baseweb="select"] > div {
    border-radius: 3px !important;
    background: var(--bg) !important;
    border: 1px solid var(--line-bright) !important;
    letter-spacing: 2px !important;
}
section[data-testid="stSidebar"] div[data-testid="stSelectbox"]:first-of-type div[data-baseweb="select"] span {
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--text-1) !important;
}

/* ── Pip bar glows with theme accent ──────────────────────────────── */
.exp-pip.active {
    background: var(--green) !important;
    box-shadow: 0 0 6px var(--green-glow) !important;
}

/* ── Regime banner accent line uses CSS var ───────────────────────── */
.regime-accent-line {
    position: absolute; top: 0; left: 0;
    height: 1px; width: 100%;
    background: var(--bar-grad, var(--line-2));
    opacity: 0.5;
}

/* ── Kill Streamlit's running-indicator spinner overlay on buttons ─── */
button[data-testid="baseButton-secondary"] .st-emotion-cache-ocqkz7,
button[data-testid="baseButton-primary"]   .st-emotion-cache-ocqkz7,
button[data-testid="baseButton-secondary"] > div > div,
button[data-testid="baseButton-primary"]   > div > div { display:none !important; }

/* ── Remove any default Streamlit focus ring across all buttons ─────── */
button:focus { outline: none !important; box-shadow: none !important; }
*:focus-visible { outline: none !important; box-shadow: none !important; }

/* ── Prevent the iframe overlay flash that appears on st.rerun ──────── */
[data-testid="stAppViewBlockContainer"] { will-change: auto !important; }
</style>
""", unsafe_allow_html=True)

# ── Instant button-press feedback injected into parent DOM ──────────────────
_components.html("""
<script>
(function attachCrispPress() {
  function press(e) {
    var b = e.currentTarget;
    b.style.transform = 'scale(0.97)';
    b.style.transition = 'transform 0.04s ease';
    setTimeout(function(){ b.style.transform = ''; }, 120);
  }
  function bindAll() {
    document.querySelectorAll(
      'button[data-testid="baseButton-secondary"], button[data-testid="baseButton-primary"]'
    ).forEach(function(b) {
      if (!b._crispBound) {
        b.addEventListener('mousedown', press);
        b._crispBound = true;
      }
    });
  }
  bindAll();
  var mo = new MutationObserver(bindAll);
  mo.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", height=0)


# ── Navigation Bar ──────────────────────────────────────────────────────────
_T_nav = get_theme()
_components.html(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  .nav-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 24px; height: 48px;
    background: {_T_nav['nav_bg']};
    border-bottom: 1px solid {_T_nav['nav_border']};
  }}
  .nav-wordmark {{ display: flex; align-items: center; gap: 12px; font-family: 'Barlow Condensed', sans-serif; font-size: 17px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; }}
  .nav-pip {{ width: 7px; height: 7px; border-radius: 50%; background: {_T_nav['nav_dot']}; animation: pulse 3s ease-in-out infinite; }}
  .nav-word   {{ color: {_T_nav['nav_t1']}; }}
  .nav-accent {{ color: {_T_nav['nav_t2']}; }}
  .nav-right  {{ display: flex; align-items: center; gap: 18px; }}
  .nav-status {{ display: flex; align-items: center; gap: 8px; font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 600; letter-spacing: 2px; color: {_T_nav['nav_t2']}; text-transform: uppercase; }}
  .nav-dot {{ width: 5px; height: 5px; border-radius: 50%; background: {_T_nav['nav_dot']}; animation: pulse 2s ease-in-out infinite; }}
  .nav-divider {{ width: 1px; height: 16px; background: {_T_nav['nav_border']}; }}
  #live-clock {{ color: {_T_nav['nav_clock']}; font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1px; }}
  #cdown-pill {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 1px; color: {_T_nav['nav_t2']}; border: 1px solid {_T_nav['nav_border']}; border-radius: 20px; padding: 3px 10px; }}
  @keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
</style>
<div class="nav-bar">
  <div class="nav-wordmark">
    <span class="nav-pip"></span>
    <span><span class="nav-word">GEX</span><span class="nav-accent">RADAR</span></span>
  </div>
  <div class="nav-right">
    <div class="nav-status">
      <div class="nav-dot"></div>
      LIVE
    </div>
    <div class="nav-divider"></div>
    <span id="live-clock">--:--:--</span>
  </div>
</div>
<script>
  function updateClock() {{
    var now = new Date();
    var est = new Date(now.toLocaleString("en-US", {{timeZone: "America/New_York"}}));
    var h = String(est.getHours()).padStart(2,'0');
    var m = String(est.getMinutes()).padStart(2,'0');
    var s = String(est.getSeconds()).padStart(2,'0');
    document.getElementById('live-clock').textContent = h + ':' + m + ':' + s;
  }}
  updateClock(); setInterval(updateClock, 1000);
  var TOTAL = {AUTO_REFRESH_SECONDS}, secs = TOTAL;
  function tick() {{ secs = secs > 0 ? secs-1 : 0; var p = window.parent.document.getElementById('gex-cdown'); if(p) p.textContent = secs; }}
  setInterval(tick, 1000);
  function attachObserver() {{
    var signal = window.parent.document.getElementById('gex-refresh-signal');
    if (!signal) {{ setTimeout(attachObserver, 500); return; }}
    new MutationObserver(function() {{ secs = TOTAL; var p = window.parent.document.getElementById('gex-cdown'); if(p) p.textContent = secs; }}).observe(signal, {{ childList:true, characterData:true, subtree:true }});
  }}
  attachObserver();
</script>
""", height=50, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────────────────
def _d1d2(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    return d1, d1 - sigma*math.sqrt(T)

def bs_price(S, K, T, r, q, sigma, flag):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    if flag == "C":
        return S*math.exp(-q*T)*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    return K*math.exp(-r*T)*norm.cdf(-d2) - S*math.exp(-q*T)*norm.cdf(-d1)

def bs_gamma(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return math.exp(-q*T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_delta(S, K, T, r, q, sigma, flag):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return math.exp(-q*T)*norm.cdf(d1) if flag=="C" else -math.exp(-q*T)*norm.cdf(-d1)

def bs_vega(S, K, T, r, q, sigma):
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return S*math.exp(-q*T)*norm.pdf(d1)*math.sqrt(T)

def bs_charm(S, K, T, r, q, sigma, flag):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    c = -math.exp(-q*T)*norm.pdf(d1)*(2*(r-q)*T - d2*sigma*math.sqrt(T))/(2*T*sigma*math.sqrt(T))
    return c - q*math.exp(-q*T)*norm.cdf(d1) if flag=="C" else c + q*math.exp(-q*T)*norm.cdf(-d1)

def bs_vanna(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return -math.exp(-q*T)*norm.pdf(d1)*d2/sigma

def bs_vomma(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return bs_vega(S,K,T,r,q,sigma)*d1*d2/sigma

def bs_zomma(S, K, T, r, q, sigma):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if np.isnan(d1): return 0.0
    return bs_gamma(S,K,T,r,q,sigma)*(d1*d2-1)/sigma

def implied_vol(market_price, S, K, T, r, q, flag):
    if T <= 0 or market_price <= 0: return np.nan
    intrinsic = max(0.0, (S-K) if flag=="C" else (K-S))
    if market_price <= intrinsic + 1e-4: return np.nan
    try:
        iv = brentq(lambda v: bs_price(S,K,T,r,q,v,flag) - market_price,
                    1e-5, 10.0, xtol=1e-6, maxiter=200)
        return iv if 0.005 < iv < 5.0 else np.nan
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH — CBOE Public API (no key required)
# ─────────────────────────────────────────────────────────────────────────────
import requests as _requests

def _cboe_get(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.cboe.com/",
        "Origin": "https://www.cboe.com",
    }
    r = _requests.get(url, headers=headers, timeout=20)
    if not r.ok:
        raise RuntimeError(f"CBOE {url} returned HTTP {r.status_code}: {r.text[:200]}")
    return r.json()

def get_spot(ticker: str) -> float:
    key = f"_spot_{ticker}"
    now = datetime.datetime.utcnow()
    cached = st.session_state.get(key)
    if cached and (now - cached["ts"]).total_seconds() < 58:
        return cached["val"]
    symbol = ticker.upper()
    CBOE_INDICES = {"SPX", "NDX", "RUT", "VIX", "XSP", "SPXW"}
    _url = (f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"
            if symbol in CBOE_INDICES else
            f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json")
    data = _cboe_get(_url)
    val  = float(data["data"]["current_price"])
    st.session_state[key] = {"val": val, "ts": now}
    return val


def _get_chain(ticker: str) -> dict:
    key = f"_chain_{ticker}"
    now = datetime.datetime.utcnow()
    cached = st.session_state.get(key)
    if cached and (now - cached["ts"]).total_seconds() < 58:
        return cached["data"]
    symbol = ticker.upper()
    CBOE_INDICES = {"SPX", "NDX", "RUT", "VIX", "XSP", "SPXW"}
    if symbol in CBOE_INDICES:
        urls = [f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json"]
    else:
        urls = [
            f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json",
            f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{symbol}.json",
        ]
    for url in urls:
        try:
            data = _cboe_get(url)
            if data.get("data", {}).get("options"):
                st.session_state[key] = {"data": data, "ts": now}
                return data
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch options chain for {ticker} from CBOE")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: DYNAMIC ES / NQ CONVERSION RATIO
# ─────────────────────────────────────────────────────────────────────────────
if "es_spy_ratio" not in st.session_state:
    st.session_state.es_spy_ratio = None
if "nq_qqq_ratio" not in st.session_state:
    st.session_state.nq_qqq_ratio = None

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_yahoo_price(ticker: str) -> float:
    """Fetch a spot/futures price from Yahoo Finance query1 API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1m&range=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    r = _requests.get(url, headers=headers, timeout=10)
    if not r.ok:
        raise RuntimeError(f"Yahoo {ticker} HTTP {r.status_code}")
    data = r.json()
    meta = data["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice") or meta.get("previousClose")
    if not price:
        raise RuntimeError(f"No price in Yahoo response for {ticker}")
    return float(price)

def get_es_spy_ratio(spy_spot: float) -> float:
    try:
        es_price = _fetch_yahoo_price("ES=F")
        ratio = es_price / spy_spot
        st.session_state.es_spy_ratio = ratio
        return ratio
    except Exception:
        pass
    if st.session_state.es_spy_ratio is not None:
        return st.session_state.es_spy_ratio
    try:
        spx_spot = get_spot("SPX")
        ratio = spx_spot / spy_spot
        st.session_state.es_spy_ratio = ratio
        return ratio
    except Exception:
        pass
    return 10.0

def get_nq_qqq_ratio(qqq_spot: float) -> float:
    try:
        nq_price = _fetch_yahoo_price("NQ=F")
        ratio = nq_price / qqq_spot
        st.session_state.nq_qqq_ratio = ratio
        return ratio
    except Exception:
        pass
    if st.session_state.nq_qqq_ratio is not None:
        return st.session_state.nq_qqq_ratio
    try:
        ndx_spot = get_spot("NDX")
        ratio = ndx_spot / qqq_spot
        st.session_state.nq_qqq_ratio = ratio
        return ratio
    except Exception:
        pass
    return 42.0


def _parse_cboe_chain(data, spot, max_expirations=4):
    options = data["data"].get("options", [])
    today   = datetime.date.today()

    def dte(e):
        return (datetime.datetime.strptime(e, "%Y-%m-%d").date() - today).days

    def parse_symbol(sym):
        import re
        m = re.search(r'(\d{6})([CP])(\d{8})$', sym)
        if not m:
            return None, None, None
        date_str, flag, strike_str = m.group(1), m.group(2), m.group(3)
        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        strike = int(strike_str) / 1000.0
        return expiry, flag, strike

    by_exp = {}
    for opt in options:
        sym = opt.get("option", "")
        expiry, flag, strike = parse_symbol(sym)
        if not expiry or strike is None:
            continue
        by_exp.setdefault(expiry, []).append({**opt, "_expiry": expiry, "_flag": flag, "_strike": strike})

    sorted_exps = sorted(
        [e for e in by_exp if dte(e) >= 0],
        key=dte
    )[:max_expirations]

    result = {}
    for exp in sorted_exps:
        rows = []
        for opt in by_exp[exp]:
            rows.append({
                "strike":        opt["_strike"],
                "option_type":   opt["_flag"],
                "open_interest": float(opt.get("open_interest", 0) or 0),
                "volume":        float(opt.get("volume", 0) or 0),
                "bid":           float(opt.get("bid", 0) or 0),
                "ask":           float(opt.get("ask", 0) or 0),
                "iv":            float(opt.get("iv", 0) or 0),
            })
        if rows:
            result[exp] = pd.DataFrame(rows)
    return result, sorted_exps


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS CHAIN
# ─────────────────────────────────────────────────────────────────────────────
def _process_chain(chain_df, spot, T, r, q, exp, days):
    rows = []

    _atm_mask = (
        (chain_df["strike"] >= spot * 0.98) &
        (chain_df["strike"] <= spot * 1.02)
    )
    _atm_iv_vals = []
    for _, _row in chain_df[_atm_mask].iterrows():
        _iv_r = float(_row.get("iv", 0) or 0)
        if _iv_r > 0.05:
            _atm_iv_vals.append(_iv_r)
    _atm_iv_base = float(np.median(_atm_iv_vals)) if _atm_iv_vals else 0.20

    for _, row in chain_df.iterrows():
        flag = str(row["option_type"]) if "option_type" in row.index else "C"
        K    = float(row["strike"]        if "strike"        in row.index else 0)
        oi   = float(row["open_interest"] if "open_interest" in row.index else 0)
        vol  = float(row["volume"]        if "volume"        in row.index else 0)
        bid  = float(row["bid"]           if "bid"           in row.index else 0)
        ask  = float(row["ask"]           if "ask"           in row.index else 0)
        iv_r = float(row["iv"]            if "iv"            in row.index else 0)

        if K <= 0:
            continue

        dist_pct = abs(K - spot) / spot
        if dist_pct > 0.08:
            continue

        if oi < 100:
            continue

        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        else:
            mid = 0.0

        iv = np.nan

        if mid > 0.05:
            iv = implied_vol(mid, spot, K, T, r, q, flag)

        if np.isnan(iv) or iv <= 0.005:
            if iv_r > 0.05:
                iv = iv_r
            elif mid > 0.05:
                iv = _atm_iv_base * (1.0 + dist_pct * 0.5)
            else:
                continue

        iv = min(iv, 1.5)

        gamma = bs_gamma(spot, K, T, r, q, iv)
        delta = bs_delta(spot, K, T, r, q, iv, flag)
        vega  = bs_vega(spot, K, T, r, q, iv)
        charm = bs_charm(spot, K, T, r, q, iv, flag)
        vanna = bs_vanna(spot, K, T, r, q, iv)
        vomma = bs_vomma(spot, K, T, r, q, iv)
        zomma = bs_zomma(spot, K, T, r, q, iv)

        gex_oi  = gamma * oi  * 100 * (spot ** 2) * 0.01 / 1e9
        gex_vol = gamma * vol * 100 * (spot ** 2) * 0.01 / 1e9

        rows.append({
            "strike": K, "expiry": exp, "dte": days, "flag": flag,
            "open_interest": oi, "volume": vol, "last_price": mid,
            "bid": bid, "ask": ask,
            "iv": iv, "delta": delta, "gamma": gamma,
            "vega": vega, "charm": charm, "vanna": vanna,
            "vomma": vomma, "zomma": zomma,
            "call_gex":     gex_oi  if flag == "C" else 0.0,
            "put_gex":      -gex_oi if flag == "P" else 0.0,
            "call_gex_vol": gex_vol if flag == "C" else 0.0,
            "put_gex_vol":  -gex_vol if flag == "P" else 0.0,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE GEX
# ─────────────────────────────────────────────────────────────────────────────
def compute_gex(ticker: str, n_exp: int = 4):
    r = RISK_FREE_RATE
    q = DIV_YIELD.get(ticker.upper(), 0.0)
    raw   = _get_chain(ticker)
    spot  = get_spot(ticker)
    chains, exps = _parse_cboe_chain(raw, spot, max_expirations=n_exp)
    today = datetime.date.today()
    frames = []
    for exp in exps:
        if exp not in chains:
            continue
        days = (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        T    = max(days / 365.0, 1/365.0)
        df   = _process_chain(chains[exp], spot, T, r, q, exp, days)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), spot, exps
    return pd.concat(frames, ignore_index=True), spot, exps


def fetch_options_data_heatmap(ticker: str):
    r = RISK_FREE_RATE
    q = DIV_YIELD.get(ticker.upper(), 0.0)
    raw   = _get_chain(ticker)
    spot  = get_spot(ticker)
    today = datetime.date.today()
    chains, exps = _parse_cboe_chain(raw, spot, max_expirations=12)
    frames = []
    for exp in exps:
        if exp not in chains:
            continue
        days = (datetime.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if days > 90:
            continue
        T = max(days / 365.0, 1/365.0)
        df = _process_chain(chains[exp], spot, T, r, q, exp, days)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), spot
    return pd.concat(frames, ignore_index=True), spot


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _base_layout(title="", h=480):
    T = get_theme()
    return dict(
        title=dict(text=title, font=dict(family="JetBrains Mono", size=11, color=T["chart_t3"]),
                   x=0.01, xanchor="left", y=0.97, yanchor="top"),
        paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_bg"],
        font=dict(family="JetBrains Mono", size=10, color=T["chart_t2"]),
        height=h, margin=dict(l=48, r=20, t=36, b=36),
        xaxis=dict(gridcolor=T["chart_line"], zerolinecolor=T["chart_line2"],
                   tickfont=dict(size=9), showgrid=True),
        yaxis=dict(gridcolor=T["chart_line"], zerolinecolor=T["chart_line2"],
                   tickfont=dict(size=9), showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hoverlabel=dict(bgcolor=T["chart_hover"], bordercolor=T["chart_hover_border"],
                        font=dict(color=T["chart_hover_text"], size=10, family="JetBrains Mono")),
    )


def bar_layout(df, spot, mode="GEX"):
    T = get_theme()
    lo, hi = spot * 0.92, spot * 1.08
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    if mode in ("GEX", "VOL"):
        c_col = "call_gex" if mode == "GEX" else "call_gex_vol"
        p_col = "put_gex"  if mode == "GEX" else "put_gex_vol"
        agg = df.groupby("strike").agg({c_col: "sum", p_col: "sum"}).reset_index()
        agg["net"] = agg[c_col] + agg[p_col]
        colors = [T["bar_pos"] if v >= 0 else T["bar_neg"] for v in agg["net"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["strike"], y=agg["net"], marker_color=colors, name="Net GEX",
                             hovertemplate="Strike: %{x}<br>GEX: %{y:.3f}B<extra></extra>"))
        fig.add_vline(x=spot, line_color=T["spot_line"], line_width=1,
                      annotation_text=f"SPOT {spot:.2f}", annotation_font=dict(size=9, color=T["spot_line"]))
        layout = _base_layout(f"NET {'GEX' if mode=='GEX' else 'VOL-GEX'} BY STRIKE", h=420)
        fig.update_layout(**layout)
    elif mode == "OI":
        calls = df[df["flag"]=="C"].groupby("strike")["open_interest"].sum().reset_index()
        puts  = df[df["flag"]=="P"].groupby("strike")["open_interest"].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=calls["strike"], y=calls["open_interest"], name="Calls", marker_color=T["bar_pos"]))
        fig.add_trace(go.Bar(x=puts["strike"],  y=-puts["open_interest"], name="Puts",  marker_color=T["bar_neg"]))
        fig.add_vline(x=spot, line_color=T["spot_line"], line_width=1)
        layout = _base_layout("OPEN INTEREST BY STRIKE", h=420)
        layout["barmode"] = "relative"
        fig.update_layout(**layout)
    else:
        greek = mode.lower()
        if greek not in df.columns:
            greek = "gamma"
        agg = df.groupby("strike")[greek].sum().reset_index()
        colors = [T["bar_pos"] if v >= 0 else T["bar_neg"] for v in agg[greek]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=agg["strike"], y=agg[greek], marker_color=colors, name=mode))
        fig.add_vline(x=spot, line_color=T["spot_line"], line_width=1)
        fig.update_layout(**_base_layout(f"{mode} BY STRIKE", h=420))
    return fig


def cumulative_gex_chart(df, spot):
    T = get_theme()
    lo, hi = spot * 0.92, spot * 1.08
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()
    agg = df.groupby("strike").agg({"call_gex": "sum", "put_gex": "sum"}).reset_index()
    agg["net"] = agg["call_gex"] + agg["put_gex"]
    agg = agg.sort_values("strike")
    agg["cumgex"] = agg["net"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["strike"], y=agg["cumgex"], mode="lines",
                             line=dict(color=T["blue"], width=2),
                             fill="tozeroy", fillcolor="rgba(74,159,255,0.07)", name="Cum GEX"))
    fig.add_vline(x=spot, line_color=T["spot_line"], line_width=1)
    fig.add_hline(y=0, line_color=T["chart_line2"], line_width=1)
    fig.update_layout(**_base_layout("CUMULATIVE GEX", h=280))
    return fig


def iv_smile_chart(df, spot):
    T = get_theme()
    fig = go.Figure()
    for exp, grp in df.groupby("expiry"):
        grp = grp.sort_values("strike")
        days = grp["dte"].iloc[0] if "dte" in grp.columns else 0
        fig.add_trace(go.Scatter(x=grp["strike"], y=grp["iv"]*100, mode="lines+markers",
                                 name=f"{exp} ({days}d)", marker=dict(size=4)))
    fig.add_vline(x=spot, line_color=T["spot_line"], line_width=1)
    layout = _base_layout("IV SMILE", h=360)
    layout["yaxis"]["title"] = "IV (%)"
    fig.update_layout(**layout)
    return fig


def gex_heatmap(df, spot):
    T = get_theme()
    if df.empty:
        return go.Figure()
    df = df.copy()
    df["net_gex"] = df["call_gex"] + df["put_gex"]
    pivot = df.pivot_table(index="strike", columns="expiry", values="net_gex", aggfunc="sum").sort_index()
    z = pivot.values
    zmax = max(abs(z.min()), abs(z.max()), 0.001)
    fig = go.Figure(go.Heatmap(z=z, x=list(pivot.columns), y=list(pivot.index),
                               colorscale=T["heat_colorscale"], zmid=0, zmin=-zmax, zmax=zmax,
                               colorbar=dict(title="GEX($B)", tickfont=dict(size=8))))
    fig.add_hline(y=spot, line_color=T["spot_line"], line_width=1,
                  annotation_text=f"SPOT {spot:.1f}", annotation_font=dict(size=9))
    layout = _base_layout("GEX HEATMAP", h=480)
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig


def greek_surface(df, spot, greek="gamma"):
    T = get_theme()
    if df.empty or greek not in df.columns:
        return go.Figure()
    pivot = df.pivot_table(index="strike", columns="expiry", values=greek, aggfunc="sum").sort_index()
    fig = go.Figure(go.Surface(z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                               colorscale=T["surface_colorscale"]))
    layout = _base_layout(f"{greek.upper()} SURFACE", h=520)
    layout["scene"] = dict(
        xaxis=dict(title="Expiry", backgroundcolor=T["chart_bg"], gridcolor=T["chart_line"]),
        yaxis=dict(title="Strike", backgroundcolor=T["chart_bg"], gridcolor=T["chart_line"]),
        zaxis=dict(title=greek.capitalize(), backgroundcolor=T["chart_bg"]),
        bgcolor=T["chart_bg"],
    )
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# KEY LEVELS + REGIME
# ─────────────────────────────────────────────────────────────────────────────
def compute_key_levels(df, spot):
    if df.empty:
        return {}
    df = df.copy()
    df["net_gex"] = df["call_gex"] + df["put_gex"]
    agg = df.groupby("strike").agg(
        net_gex=("net_gex","sum"), total_oi=("open_interest","sum"),
        total_vega=("vega","sum"), total_charm=("charm","sum"),
    ).reset_index()
    agg_s = agg.sort_values("strike")
    cum = agg_s["net_gex"].cumsum().values
    strikes = agg_s["strike"].values
    flip = None
    for i in range(len(cum)-1):
        if cum[i]*cum[i+1] <= 0:
            flip = strikes[i]; break
    max_gex_row = agg.loc[agg["net_gex"].abs().idxmax()]
    max_oi_row  = agg.loc[agg["total_oi"].idxmax()]
    put_oi  = df[df["flag"]=="P"].groupby("strike")["open_interest"].sum()
    call_oi = df[df["flag"]=="C"].groupby("strike")["open_interest"].sum()
    hvl_row = agg.loc[agg["total_vega"].abs().idxmax()]
    charm_pos = agg[agg["total_charm"]>0]["strike"].values
    charm_neg = agg[agg["total_charm"]<0]["strike"].values
    return {
        "spot": spot, "gex_flip": flip,
        "max_gex_k": max_gex_row["strike"], "max_gex_val": max_gex_row["net_gex"],
        "max_oi_k": max_oi_row["strike"], "max_oi_val": max_oi_row["total_oi"],
        "max_put_k":  put_oi.idxmax()  if not put_oi.empty  else None,
        "max_call_k": call_oi.idxmax() if not call_oi.empty else None,
        "hvl": hvl_row["strike"],
        "charm_support": float(charm_neg.min()) if len(charm_neg) else None,
        "charm_resist":  float(charm_pos.max()) if len(charm_pos) else None,
    }


def regime_label(kl):
    flip = kl.get("gex_flip")
    spot = kl.get("spot", 0)
    if flip is None:
        return "UNKNOWN", "NEUTRAL"
    return ("POSITIVE GEX", "BULLISH") if spot > flip else ("NEGATIVE GEX", "BEARISH")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
ASSET_LIST = [
    "-- ETFs --", "SPY", "QQQ", "IWM", "GLD", "SLV", "TLT", "XLF", "XLE", "IBIT",
    "-- Mega-Cap --", "AAPL", "NVDA", "TSLA", "AMZN", "MSFT", "META", "GOOGL",
    "-- Indices --", "SPX", "NDX", "RUT",
]

with st.sidebar:
    st.selectbox("Theme", list(THEMES.keys()), key="ui_theme")
    st.markdown('<div class="sb-section">ASSET</div>', unsafe_allow_html=True)
    asset = st.selectbox("Asset", ASSET_LIST, key="asset_choice")
    if asset.startswith("--"):
        asset = "SPY"
        st.session_state.asset_choice = "SPY"

    st.markdown('<div class="sb-section">EXPIRATIONS</div>', unsafe_allow_html=True)
    n_exp = st.slider("", 1, 8, 4, key="n_exp", label_visibility="collapsed")

    st.markdown('<div class="sb-section">VIEW MODE</div>', unsafe_allow_html=True)
    mode_options = ["GEX", "VOL", "OI", "DELTA", "VEGA", "CHARM", "VANNA"]
    radar_mode = st.radio("Mode", mode_options, key="radar_mode", label_visibility="collapsed")

    st.markdown('<div class="sb-section">DATA</div>', unsafe_allow_html=True)
    if st.button("REFRESH"):
        for k in list(st.session_state.keys()):
            if k.startswith("_spot_") or k.startswith("_chain_"):
                del st.session_state[k]
        st.rerun()

    st.markdown('<div class="sb-section">MARKET PULSE</div>', unsafe_allow_html=True)
    watch_html = ""
    for wt in ["SPY", "QQQ", "IWM", "TLT", "GLD"]:
        try:
            wp = get_spot(wt)
            watch_html += f'<div class="m-tile"><span class="m-label">{wt}</span><span class="m-value">{wp:.2f}</span></div>'
        except Exception:
            watch_html += f'<div class="m-tile"><span class="m-label">{wt}</span><span class="m-value">--</span></div>'
    st.markdown(f'<div class="sb-group">{watch_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
ticker = st.session_state.get("asset_choice", "SPY")
if ticker.startswith("--"):
    ticker = "SPY"

with st.spinner(f"Loading {ticker} options..."):
    try:
        df, spot, exps = compute_gex(ticker, n_exp=st.session_state.get("n_exp", 4))
        error_msg = None
    except Exception as e:
        df, spot, exps = pd.DataFrame(), 0.0, []
        error_msg = str(e)

kl = compute_key_levels(df, spot) if not df.empty else {}
regime, bias = regime_label(kl)
T = get_theme()
bar_color = T["bar_pos"] if bias == "BULLISH" else T["bar_neg"] if bias == "BEARISH" else T["amber"]

st.markdown(f"""
<div class="regime" style="--bar-grad: linear-gradient(90deg, {bar_color}, transparent);">
  <div class="regime-accent-line"></div>
  <div>
    <div class="regime-meta">{ticker} &middot; SPOT {spot:.2f}</div>
    <div class="regime-state" style="color:{bar_color}">{regime}</div>
  </div>
  <div class="regime-right">
    <div class="regime-bias-label">DEALER BIAS</div>
    <div class="regime-bias-value" style="color:{bar_color}">{bias}</div>
  </div>
</div>
""", unsafe_allow_html=True)

if error_msg:
    st.error(f"Data error: {error_msg}")
elif df.empty:
    st.warning("No options data returned. Markets may be closed or ticker unsupported.")
else:
    st.markdown('<div class="sec-head">KEY LEVELS</div>', unsafe_allow_html=True)
    kl_items = [
        ("GEX Flip",       f"{kl['gex_flip']:.1f}"   if kl.get('gex_flip')       else "--"),
        ("Max GEX Strike", f"{kl['max_gex_k']:.1f}"  if kl.get('max_gex_k')      else "--"),
        ("Max GEX Value",  f"{kl['max_gex_val']:.3f}B" if kl.get('max_gex_val')  else "--"),
        ("Max OI Strike",  f"{kl['max_oi_k']:.1f}"   if kl.get('max_oi_k')       else "--"),
        ("Max Put OI",     f"{kl['max_put_k']:.1f}"  if kl.get('max_put_k')      else "--"),
        ("Max Call OI",    f"{kl['max_call_k']:.1f}" if kl.get('max_call_k')     else "--"),
        ("HVL (Vega)",     f"{kl['hvl']:.1f}"        if kl.get('hvl')            else "--"),
        ("Charm Support",  f"{kl['charm_support']:.1f}" if kl.get('charm_support') else "--"),
        ("Charm Resist",   f"{kl['charm_resist']:.1f}"  if kl.get('charm_resist')  else "--"),
    ]
    kl_html = '<div class="kl-panel"><div class="kl-header">LEVELS</div>'
    for name, val in kl_items:
        kl_html += f'<div class="kl-row"><span class="kl-name">{name}</span><span class="kl-val">{val}</span></div>'
    kl_html += '</div>'
    st.markdown(kl_html, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">EXPOSURE</div>', unsafe_allow_html=True)
    st.plotly_chart(bar_layout(df, spot, mode=st.session_state.get("radar_mode","GEX")), use_container_width=True)
    st.plotly_chart(cumulative_gex_chart(df, spot), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sub-head">IV SMILE</div>', unsafe_allow_html=True)
        st.plotly_chart(iv_smile_chart(df, spot), use_container_width=True)
    with col2:
        st.markdown('<div class="sub-head">GEX HEATMAP</div>', unsafe_allow_html=True)
        try:
            df_h, spot_h = fetch_options_data_heatmap(ticker)
            st.plotly_chart(gex_heatmap(df_h, spot_h), use_container_width=True)
        except Exception as ex:
            st.warning(f"Heatmap unavailable: {ex}")

    st.markdown('<div class="sec-head">GREEK SURFACE</div>', unsafe_allow_html=True)
    g_choice = st.selectbox("Greek", ["gamma","vega","delta","charm","vanna","vomma","zomma"],
                             key="greek_surface_choice", label_visibility="collapsed")
    st.plotly_chart(greek_surface(df, spot, greek=g_choice), use_container_width=True)

    with st.expander("RAW CHAIN DATA"):
        display_cols = [c for c in ["strike","expiry","dte","flag","open_interest","volume",
                                    "bid","ask","iv","delta","gamma","vega","call_gex","put_gex"]
                        if c in df.columns]
        st.dataframe(df[display_cols].sort_values(["expiry","strike"]).reset_index(drop=True),
                     use_container_width=True, hide_index=True)
        st.download_button("EXPORT CSV", df.to_csv(index=False).encode(), f"{ticker}_gex.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div id="gex-refresh-signal" style="display:none">0</div>', unsafe_allow_html=True)
time.sleep(AUTO_REFRESH_SECONDS)
st.rerun()
