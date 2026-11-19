from datetime import date, timedelta
from html import escape as escape_html
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from database import (
    get_inventory_df,
    get_sales_df,
    get_settings,
    init_db,
    record_sale,
    save_settings,
    seed_sample_data,
    seed_indian_grocery_demo_data,
    upsert_inventory_item,
)
from gemini_assistant import (
    generate_shop_advice_stream,
    requested_chart_type,
    requests_chart,
)
from indian_formatting import format_indian_number, format_inr
from profit_analysis import summarize_sku_profit


INK = "#1E1D1B"
MUTED = "#5C5852"
PAPER = "#FCF9F2"
CARD_BG = "#FFFDF9"
CORAL = "#E05A47"
TEAL = "#2A7B76"
MUSTARD = "#D99B26"
LAVENDER = "#6C63B5"


def indian_currency_tick_options(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return {}

    lower, upper = float(numeric.min()), float(numeric.max())
    ticks = [lower] if lower == upper else np.linspace(lower, upper, 5)
    return {
        "tickmode": "array",
        "tickvals": ticks,
        "ticktext": [format_inr(value, decimals=0) for value in ticks],
    }


def make_doodle_layout(title: str, y_title: str = "", x_title: str = "") -> dict:
    return dict(
        title=dict(
            text=title,
            font=dict(family="Patrick Hand, Segoe Print, Comic Sans MS, cursive", size=19, color=INK),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=PAPER,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=12),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=INK,
            bordercolor=INK,
            font=dict(family="Inter, sans-serif", size=11, color="#FFFDF9"),
        ),
        margin=dict(l=18, r=18, t=48, b=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(family="Patrick Hand, Segoe Print, Comic Sans MS, cursive", size=14, color=INK),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(family="Patrick Hand, Segoe Print, Comic Sans MS, cursive", size=14)),
            showgrid=True,
            gridcolor="rgba(30, 29, 27, 0.07)",
            griddash="dot",
            showline=True,
            linecolor=INK,
            linewidth=2,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=MUTED),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(family="Patrick Hand, Segoe Print, Comic Sans MS, cursive", size=14)),
            showgrid=True,
            gridcolor="rgba(30, 29, 27, 0.09)",
            griddash="dash",
            zeroline=False,
            showline=True,
            linecolor=INK,
            linewidth=2,
            tickfont=dict(family="JetBrains Mono, monospace", size=11, color=MUTED),
        ),
    )


def _legacy_doodle_notebook_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Patrick+Hand&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --paper: #FCF9F2;
            --card: #FFFDF9;
            --ink: #1E1D1B;
            --muted: #5C5852;
            --coral: #E05A47;
            --teal: #2A7B76;
            --mustard: #D99B26;
            --mint-wash: #EAF5F0;
            --peach-wash: #FFF0E8;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background-color: var(--paper);
            background-image:
                radial-gradient(circle at 0% 0%, rgba(8, 127, 115, 0.07), transparent 28rem),
                radial-gradient(circle at 100% 10%, rgba(249, 115, 85, 0.06), transparent 24rem);
            background-attachment: fixed;
        }

        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5EAE8;
        }

        h1, h2, h3 {
            font-family: 'Manrope', 'Inter', sans-serif !important;
            color: var(--ink) !important;
            letter-spacing: -0.025em !important;
        }

        h2 {
            font-size: 1.45rem !important;
        }

        h3 {
            font-size: 1.2rem !important;
        }

        .doodle-header {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(120deg, rgba(8, 127, 115, 0.96), rgba(23, 32, 51, 0.98));
            border: 1px solid rgba(23, 32, 51, 0.12);
            border-radius: 20px;
            padding: 1.25rem 1.45rem 1.15rem;
            box-shadow: 0 12px 32px rgba(23, 32, 51, 0.12);
            margin: 0.25rem 0 1.25rem;
        }

        .doodle-header:after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            right: -72px;
            top: -110px;
            border: 34px solid rgba(255, 255, 255, 0.08);
            border-radius: 50%;
        }

        .doodle-toprow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255, 255, 255, 0.78);
            margin-bottom: 0.25rem;
        }

        .doodle-title {
            position: relative;
            z-index: 1;
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.08;
            color: #FFFFFF !important;
            margin: 0;
        }

        .doodle-sub {
            position: relative;
            z-index: 1;
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem;
            color: rgba(255, 255, 255, 0.72);
            margin-top: 0.4rem;
        }

        .doodle-secret div[data-testid="stButton"] > button {
            position: relative;
            z-index: 2;
            background: rgba(255, 255, 255, 0.12) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.78rem !important;
            padding: 0.35rem 0.7rem !important;
            min-height: 2.2rem !important;
        }

        .doodle-secret div[data-testid="stButton"] > button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: none;
        }

        .note-banner {
            background: #F0F7F5;
            color: var(--ink);
            border: 1px solid #BFD9D3;
            border-left: 4px solid var(--teal);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }

        .note-banner.urgent {
            background: #FFF3EF;
            border-color: #FFD0C4;
            border-left-color: var(--coral);
            color: #8A3322;
        }

        .sketch-card {
            background: #FFFFFF;
            border: 1px solid #E5EAE8;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 5px 18px rgba(23, 32, 51, 0.06);
            margin-bottom: 0.9rem;
            transition: transform 0.16s ease, box-shadow 0.16s ease;
            container-type: inline-size;
        }

        .sketch-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr));
            gap: 0.9rem;
            align-items: stretch;
            margin-bottom: 0.9rem;
        }

        .sketch-card-grid .sketch-card {
            height: 100%;
            min-width: 0;
            margin-bottom: 0;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));
            gap: 0.8rem;
            margin: 0.35rem 0 1.15rem;
        }

        .insight-card {
            min-width: 0;
            padding: 0.9rem 1rem;
            background: #FFFFFF;
            border: 1px solid #E5EAE8;
            border-radius: 14px;
            box-shadow: 0 5px 18px rgba(23, 32, 51, 0.05);
        }

        .insight-label {
            color: var(--teal);
            font-family: 'Inter', sans-serif;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            line-height: 1.1;
        }

        .insight-value {
            margin-top: 0.2rem;
            color: var(--ink);
            font-family: 'Manrope', 'Inter', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }

        .insight-detail {
            margin-top: 0.25rem;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .sketch-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 9px 24px rgba(23, 32, 51, 0.09);
        }

        .sketch-card.alert {
            border-top: 3px solid var(--coral);
        }

        .sketch-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.055em;
            color: var(--teal);
            line-height: 1.1;
        }

        .sketch-val {
            font-family: 'Manrope', 'Inter', sans-serif;
            font-variant-numeric: tabular-nums;
            font-size: clamp(0.72rem, 9.5cqw, 1.65rem);
            font-weight: 800;
            color: var(--ink);
            margin-top: 0.2rem;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }

        .sketch-sub {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 0.25rem;
        }

        [data-testid="stPlotlyChart"] > div {
            background: #FFFFFF;
            border: 1px solid #E5EAE8;
            border-radius: 14px;
            padding: 0.45rem;
            box-shadow: 0 6px 20px rgba(23, 32, 51, 0.06);
        }

        [data-testid="stDataFrame"] > div {
            border: 1px solid #E5EAE8 !important;
            border-radius: 12px !important;
            box-shadow: 0 5px 18px rgba(23, 32, 51, 0.05);
        }

        div[data-testid="stForm"] {
            background: #FFFFFF;
            border: 1px solid #E5EAE8;
            border-radius: 14px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 5px 18px rgba(23, 32, 51, 0.05);
        }

        .stButton > button, .stFormSubmitButton > button {
            background: var(--teal) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--teal) !important;
            border-radius: 9px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 700 !important;
            padding: 0.45rem 1rem !important;
            box-shadow: 0 3px 8px rgba(8, 127, 115, 0.16) !important;
            transition: transform 0.1s ease, filter 0.12s ease !important;
        }

        .stButton > button:hover, .stFormSubmitButton > button:hover {
            background: #056B62 !important;
            transform: translateY(-1px);
            filter: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            flex-wrap: wrap;
            border-bottom: 1px solid #DDE4E2;
            padding-bottom: 0.3rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.88rem !important;
            font-weight: 700 !important;
            background: transparent !important;
            color: var(--muted) !important;
            border: 0 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 0.55rem 0.9rem !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: #EEF4F2 !important;
        }

        .stTabs [aria-selected="true"] {
            background: #FFFFFF !important;
            color: var(--teal) !important;
            border-bottom: 3px solid var(--teal);
        }

        [data-testid="stDataFrame"] thead tr th {
            background: #F3F6F5 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.78rem;
            color: var(--ink) !important;
        }

        [data-testid="stPlotlyChart"] {
            filter: none;
        }

        @media (max-width: 768px) {
            .doodle-header {
                padding: 0.9rem 1rem;
            }
            .doodle-title {
                font-size: 1.7rem;
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 1rem !important;
                padding: 0.25rem 0.65rem !important;
            }
        }

        /* Original notebook skin */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .stApp {
            background-color: #FCF9F2;
            background-image:
                radial-gradient(rgba(30,29,27,0.16) 1.1px, transparent 1.1px),
                linear-gradient(135deg, rgba(42,123,118,0.055), transparent 38%),
                linear-gradient(315deg, rgba(224,90,71,0.045), transparent 42%);
            background-size: 22px 22px, 100% 100%, 100% 100%;
            isolation: isolate;
        }

        .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background-image:
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 160 160'%3E%3Cg fill='none' stroke='%232A7B76' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='34' cy='36' r='13'/%3E%3Cpath d='M27 36h14M34 29v14M58 28l10 10M68 28L58 38M112 25l8 8M120 25l-8 8M116 18v22M105 29h22'/%3E%3Cpath d='M18 105c12-12 30-12 42 0l-7 18H25zM25 123l-8 8 13-5M92 92h36v28H92zM98 98l12 9 12-9M138 112l8-10 8 10-8 10zM146 122v14M139 136h14'/%3E%3C/g%3E%3Cg fill='none' stroke='%23E05A47' stroke-width='1.2' stroke-linecap='round'%3E%3Cpath d='M8 72c14-10 28-10 42 0M104 145c10-8 22-8 32 0M128 68c10-8 18-8 26 0'/%3E%3C/g%3E%3C/svg%3E");
            background-repeat: repeat;
            background-size: 160px 160px;
            opacity: 0.38;
        }
        h1, h2, h3 {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive !important;
            color: #1E1D1B !important;
            letter-spacing: 0.01em !important;
        }
        .doodle-header {
            background: linear-gradient(115deg,#FFFDF9 0%,#FFF8E9 58%,#EFF8F4 100%);
            border: 2px solid #1E1D1B;
            border-radius: 14px 255px 15px 225px / 225px 15px 255px 14px;
            padding: 1.15rem 1.4rem 1.05rem;
            box-shadow: 4px 5px 0 rgba(30,29,27,0.85);
            margin-bottom: 0.9rem;
        }
        .doodle-header:after { display: none; }
        .doodle-toprow {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 1.05rem;
            color: #2A7B76;
            text-transform: none;
            letter-spacing: 0;
        }
        .doodle-title {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E1D1B !important;
        }
        .doodle-sub {
            font-family: 'JetBrains Mono',monospace;
            color: #5C5852;
        }
        .doodle-secret div[data-testid="stButton"] > button {
            background: transparent !important;
            color: rgba(30,29,27,0.35) !important;
            border: 1.5px dashed rgba(30,29,27,0.3) !important;
            border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px !important;
            box-shadow: none !important;
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive !important;
        }
        .doodle-secret div[data-testid="stButton"] > button:hover {
            color: #E05A47 !important;
            background: rgba(224,90,71,0.08) !important;
            transform: none;
        }
        .note-banner {
            background: linear-gradient(110deg,#FFF4CC 0%,#FFF9E6 100%);
            color: #1E1D1B;
            border: 2px solid #1E1D1B;
            border-left: 2px solid #1E1D1B;
            border-radius: 18px 6px 16px 8px / 7px 16px 8px 18px;
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 1.15rem;
            box-shadow: 3px 3px 0 rgba(30,29,27,0.75);
        }
        .note-banner.urgent {
            background: linear-gradient(110deg,#FFE5DF 0%,#FFF0EA 100%);
            border-color: #E05A47;
            color: #7D1D10;
        }
        .sketch-card {
            background: linear-gradient(145deg,#FFFDF9 0%,#FFF8E9 68%,#EAF5F0 100%);
            border: 2px solid #1E1D1B;
            border-radius: 12px 240px 14px 210px / 210px 14px 240px 12px;
            box-shadow: 3px 4px 0 rgba(30,29,27,0.8);
        }
        .sketch-card.alert {
            border: 2px solid #E05A47;
            border-top: 2px solid #E05A47;
            box-shadow: 3px 4px 0 #E05A47;
        }
        .sketch-card:hover { transform: rotate(-0.6deg) translateY(-1px); }
        .sketch-label {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 1.15rem;
            text-transform: none;
            letter-spacing: 0;
            color: #2A7B76;
        }
        .sketch-val {
            font-family: 'JetBrains Mono',monospace;
            font-weight: 700;
        }
        .insight-card {
            background: linear-gradient(125deg,#FFFDF9 0%,#FFF6E5 60%,#EAF5F0 100%);
            border: 1.5px dashed #1E1D1B;
            border-radius: 14px 28px 12px 24px / 24px 12px 28px 14px;
            box-shadow: 2px 3px 0 rgba(30,29,27,0.68);
        }
        .insight-label {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 1.05rem;
            text-transform: none;
            letter-spacing: 0;
        }
        .insight-value {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive;
            font-size: 1.35rem;
        }
        [data-testid="stPlotlyChart"] > div {
            background: linear-gradient(145deg,#FFFDF9 0%,#EAF5F0 55%,#FFF0E8 100%);
            border: 2px solid #1E1D1B;
            border-radius: 14px 220px 12px 240px / 220px 12px 240px 14px;
            box-shadow: 4px 4px 0 rgba(30,29,27,0.8);
        }
        [data-testid="stDataFrame"] > div,
        div[data-testid="stForm"] {
            background: #FFFDF9;
            border: 2px solid #1E1D1B !important;
            border-radius: 14px;
            box-shadow: 3px 3px 0 rgba(30,29,27,0.75);
        }
        .stButton > button, .stFormSubmitButton > button {
            background: linear-gradient(135deg,#E05A47 0%,#F27C63 100%) !important;
            color: #FFFDF9 !important;
            border: 2px solid #1E1D1B !important;
            border-radius: 255px 18px 225px 16px / 16px 225px 18px 255px !important;
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive !important;
            font-size: 1.15rem !important;
            box-shadow: 2px 3px 0 #1E1D1B !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px dashed #1E1D1B;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Patrick Hand','Segoe Print','Comic Sans MS',cursive !important;
            font-size: 1.15rem !important;
            background: #FFFDF9 !important;
            color: #5C5852 !important;
            border: 2px solid #1E1D1B !important;
            border-radius: 12px 20px 8px 16px !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg,#F5D579 0%,#F3B775 100%) !important;
            color: #1E1D1B !important;
            border-bottom: 2px solid #1E1D1B;
            box-shadow: 2px 2px 0 #1E1D1B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_doodle_notebook_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Patrick+Hand&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --paper: #FCF9F2;
            --card: #FFFDF9;
            --ink: #1E1D1B;
            --muted: #5C5852;
            --coral: #E05A47;
            --teal: #2A7B76;
            --mint-wash: #EAF5F0;
            --peach-wash: #FFF0E8;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background-color: var(--paper);
            background-image:
                radial-gradient(rgba(30, 29, 27, 0.16) 1.1px, transparent 1.1px),
                linear-gradient(135deg, rgba(42, 123, 118, 0.055), transparent 38%),
                linear-gradient(315deg, rgba(224, 90, 71, 0.045), transparent 42%);
            background-size: 22px 22px, 100% 100%, 100% 100%;
            background-attachment: fixed;
        }
        .stApp:before { display: none !important; }

        [data-testid="stSidebar"] {
            background-color: #F6F1E6;
            border-right: 2px dashed var(--ink);
        }

        h1, h2, h3 {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive !important;
            color: var(--ink) !important;
            letter-spacing: 0.01em !important;
        }

        .st-key-doodle-header {
            background: linear-gradient(115deg, #FFFDF9 0%, #FFF8E9 58%, #EFF8F4 100%);
            border: 2px solid var(--ink) !important;
            border-radius: 14px 255px 15px 225px / 225px 15px 255px 14px;
            padding: 1.15rem 1.4rem 1.05rem !important;
            box-shadow: 4px 5px 0 rgba(30, 29, 27, 0.85);
            margin-bottom: 0.9rem;
            gap: 0.35rem;
        }
        .doodle-toprow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive;
            font-size: 1.05rem;
            color: var(--teal);
            margin-bottom: 0.25rem;
        }
        .doodle-title {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive;
            font-size: 2.2rem;
            line-height: 1.08;
            color: var(--ink) !important;
            margin: 0;
        }
        .doodle-sub {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: var(--muted);
            margin-top: 0.4rem;
        }

        .note-banner {
            background: linear-gradient(110deg, #FFF4CC 0%, #FFF9E6 100%);
            color: var(--ink);
            border: 2px solid var(--ink);
            border-radius: 18px 6px 16px 8px / 7px 16px 8px 18px;
            padding: 0.65rem 1rem;
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive;
            font-size: 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 3px 3px 0 rgba(30, 29, 27, 0.75);
        }
        .note-banner.urgent {
            background: linear-gradient(110deg, #FFE5DF 0%, #FFF0EA 100%);
            border-color: var(--coral);
            color: #7D1D10;
            box-shadow: 3px 3px 0 var(--coral);
        }

        .sketch-card-grid, .insight-grid {
            display: grid;
            gap: 0.9rem;
            margin-bottom: 0.9rem;
        }
        .sketch-card-grid { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
        .insight-grid { grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
        .sketch-card, .insight-card {
            background: linear-gradient(145deg, #FFFDF9 0%, #FFF8E9 68%, var(--mint-wash) 100%);
            border: 2px solid var(--ink);
            border-radius: 12px 240px 14px 210px / 210px 14px 240px 12px;
            box-shadow: 3px 4px 0 rgba(30, 29, 27, 0.8);
            padding: 0.95rem 1.1rem;
            transition: transform 0.12s ease;
        }
        .insight-card {
            border-style: dashed;
            border-radius: 14px 28px 12px 24px / 24px 12px 28px 14px;
        }
        .sketch-card:hover { transform: rotate(-0.6deg) translateY(-1px); }
        .sketch-card.alert {
            border-color: var(--coral);
            box-shadow: 3px 4px 0 var(--coral);
        }
        .sketch-label, .insight-label {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive;
            color: var(--teal);
            line-height: 1.1;
        }
        .sketch-label { font-size: 1.15rem; }
        .insight-label { font-size: 1.05rem; }
        .sketch-val, .insight-value {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive;
            color: var(--ink);
            font-weight: 700;
            line-height: 1.1;
        }
        .sketch-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: clamp(0.72rem, 1.6vw, 1.65rem);
            margin-top: 0.2rem;
        }
        .insight-value { font-size: 1.35rem; margin-top: 0.2rem; }
        .sketch-sub, .insight-detail {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }

        [data-testid="stPlotlyChart"] > div {
            background: linear-gradient(145deg, #FFFDF9 0%, var(--mint-wash) 55%, var(--peach-wash) 100%);
            border: 2px solid var(--ink);
            border-radius: 14px 220px 12px 240px / 220px 12px 240px 14px;
            padding: 0.4rem;
            box-shadow: 4px 4px 0 rgba(30, 29, 27, 0.8);
        }
        [data-testid="stDataFrame"] > div {
            border: 2px solid var(--ink) !important;
            border-radius: 12px;
            box-shadow: 3px 3px 0 rgba(30, 29, 27, 0.75);
        }
        [data-testid="stDataFrame"] thead tr th {
            background: linear-gradient(90deg, #EAF5F0, #FFF4DF) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stForm"] {
            background: linear-gradient(145deg, #FFFDF9 0%, #FFF9EF 62%, #F2F8F5 100%);
            border: 2px solid var(--ink);
            border-radius: 14px 230px 12px 215px / 215px 12px 230px 14px;
            padding: 1.05rem 1.2rem;
            box-shadow: 4px 4px 0 rgba(30, 29, 27, 0.8);
        }
        .stButton > button, .stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--coral) 0%, #F27C63 100%) !important;
            color: #FFFDF9 !important;
            border: 2px solid var(--ink) !important;
            border-radius: 255px 18px 225px 16px / 16px 225px 18px 255px !important;
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive !important;
            font-size: 1.15rem !important;
            box-shadow: 2px 3px 0 var(--ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: 2px dashed var(--ink);
            padding-bottom: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: 'Patrick Hand', 'Segoe Print', 'Comic Sans MS', cursive !important;
            font-size: 1.15rem !important;
            background: var(--card) !important;
            color: var(--muted) !important;
            border: 2px solid var(--ink) !important;
            border-radius: 12px 20px 8px 16px !important;
            padding: 0.3rem 0.85rem !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #F5D579 0%, #F3B775 100%) !important;
            color: var(--ink) !important;
            box-shadow: 2px 2px 0 var(--ink);
        }

        @media (max-width: 768px) {
            .st-key-doodle-header {
                padding: 0.9rem 1rem !important;
            }
            .doodle-title { font-size: 1.7rem; }
            .sketch-card-grid { grid-template-columns: 1fr 1fr; }
            .stTabs [data-baseweb="tab"] {
                font-size: 1rem !important;
                padding: 0.25rem 0.65rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_squiggle_rule() -> None:
    st.markdown(
        """
        <svg viewBox="0 0 600 10" width="100%" height="10" preserveAspectRatio="none" style="display:block; margin: 0.55rem 0 0.75rem 0;">
            <path d="M0,5 Q25,0 50,5 T100,5 T150,5 T200,5 T250,5 T300,5 T350,5 T400,5 T450,5 T500,5 T550,5 T600,5"
                  fill="none" stroke="#1E1D1B" stroke-width="1.8" stroke-linecap="round" opacity="0.65"/>
        </svg>
        """,
        unsafe_allow_html=True,
    )


def render_sketch_cards(cards: List[Tuple[str, str, str, bool]]) -> None:
    card_markup = []
    for label, value, sub, alert in cards:
        cls = "sketch-card alert" if alert else "sketch-card"
        card_markup.append(
            f'<div class="{cls}"><div class="sketch-label">{escape_html(label)}</div>'
            f'<div class="sketch-val">{escape_html(value)}</div>'
            f'<div class="sketch-sub">{escape_html(sub)}</div></div>'
        )
    st.markdown(
        f'<div class="sketch-card-grid">{"".join(card_markup)}</div>',
        unsafe_allow_html=True,
    )


def render_insight_cards(cards: List[Tuple[str, str, str]]) -> None:
    card_markup = []
    for label, value, detail in cards:
        card_markup.append(
            f'<div class="insight-card"><div class="insight-label">{escape_html(label)}</div>'
            f'<div class="insight-value">{escape_html(value)}</div>'
            f'<div class="insight-detail">{escape_html(detail)}</div></div>'
        )
    st.markdown(
        f'<div class="insight-grid">{"".join(card_markup)}</div>',
        unsafe_allow_html=True,
    )


def build_doodle_gross_sales_chart(sales_df: pd.DataFrame) -> go.Figure:
    sales = sales_df.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    sales["week_start"] = sales["sale_date"] - pd.to_timedelta(
        sales["sale_date"].dt.day_of_week, unit="D"
    )
    weekly = (
        sales.groupby("week_start", as_index=False)
        .agg(total_amount=("total_amount", "sum"), units=("quantity", "sum"))
        .sort_values("week_start")
    )
    avg_val = float(weekly["total_amount"].mean()) if not weekly.empty else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start"],
            y=weekly["total_amount"],
            mode="lines+markers",
            name="Gross Sales",
            fill="tozeroy",
            fillcolor="rgba(42, 123, 118, 0.12)",
            line=dict(color=TEAL, width=3, shape="spline", smoothing=0.55),
            marker=dict(size=8, color=CORAL, line=dict(color=INK, width=2)),
            customdata=np.column_stack(
                [weekly["units"], weekly["total_amount"].map(format_inr)]
            ),
            hovertemplate=(
                "<b>Week of %{x|%b %d}</b><br>"
                "Gross sales: %{customdata[1]}<br>"
                "Units: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    if len(weekly) >= 2:
        fig.add_hline(
            y=avg_val,
            line_dash="dash",
            line_color=MUSTARD,
            line_width=2,
            annotation_text=f"weekly avg ~ {format_inr(avg_val, decimals=0)}",
            annotation_position="top left",
            annotation_font=dict(family="Patrick Hand, Segoe Print, Comic Sans MS, cursive", size=14, color=INK),
        )
    fig.update_layout(
        **make_doodle_layout("Gross Sales Doodle · Weekly", y_title="Gross Sales (₹)")
    )
    fig.update_yaxes(**indian_currency_tick_options(weekly["total_amount"]))
    return fig


def build_doodle_gross_profit_chart(sales_df: pd.DataFrame) -> Optional[go.Figure]:
    if sales_df.empty or "unit_cost" not in sales_df.columns:
        return None

    costed_sales = sales_df.dropna(subset=["unit_cost"]).copy()
    if costed_sales.empty:
        return None

    costed_sales["gross_profit"] = (
        costed_sales["total_amount"] - costed_sales["quantity"] * costed_sales["unit_cost"]
    )
    costed_sales["sale_date"] = pd.to_datetime(costed_sales["sale_date"])
    costed_sales["week_start"] = costed_sales["sale_date"] - pd.to_timedelta(
        costed_sales["sale_date"].dt.day_of_week, unit="D"
    )
    weekly = (
        costed_sales.groupby("week_start", as_index=False)
        .agg(gross_profit=("gross_profit", "sum"), costed_units=("quantity", "sum"))
        .sort_values("week_start")
    )
    avg_profit = float(weekly["gross_profit"].mean())

    fig = go.Figure(
        go.Scatter(
            x=weekly["week_start"],
            y=weekly["gross_profit"],
            mode="lines+markers",
            name="Gross Profit",
            fill="tozeroy",
            fillcolor="rgba(224, 90, 71, 0.13)",
            line=dict(color=CORAL, width=3, shape="spline", smoothing=0.5),
            marker=dict(size=8, color=MUSTARD, line=dict(color=INK, width=2)),
            customdata=np.column_stack(
                [weekly["costed_units"], weekly["gross_profit"].map(format_inr)]
            ),
            hovertemplate=(
                "<b>Week of %{x|%b %d}</b><br>"
                "Gross profit: %{customdata[1]}<br>"
                "Costed units: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color=INK,
        line_width=1.5,
    )
    if len(weekly) >= 2:
        fig.add_hline(
            y=avg_profit,
            line_dash="dash",
            line_color=TEAL,
            line_width=2,
            annotation_text=f"weekly avg ~ {format_inr(avg_profit, decimals=0)}",
            annotation_position="top left",
            annotation_font=dict(
                family="Patrick Hand, Segoe Print, Comic Sans MS, cursive",
                size=14,
                color=INK,
            ),
        )
    fig.update_layout(
        **make_doodle_layout("Gross Profit Doodle · Weekly", y_title="Gross Profit (₹)")
    )
    fig.update_yaxes(**indian_currency_tick_options(weekly["gross_profit"]))
    return fig


def build_doodle_stock_chart(inventory_df: pd.DataFrame) -> go.Figure:
    df = inventory_df.copy()
    df["stock_gap"] = df["stock_qty"] - df["reorder_level"]
    df = (
        df.sort_values(["stock_gap", "stock_qty"], ascending=True)
        .head(14)
        .sort_values("stock_qty", ascending=True)
    )
    bar_colors = [
        CORAL if r["stock_qty"] <= r["reorder_level"] else TEAL
        for _, r in df.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["stock_qty"],
            y=df["item_name"],
            orientation="h",
            name="On Shelf",
            marker=dict(color=bar_colors, line=dict(color=INK, width=2)),
            customdata=np.column_stack([df["sku"], df["reorder_level"], df["category"]]),
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                "Group: %{customdata[2]}<br>"
                "Stock: %{x} · Reorder: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["reorder_level"],
            y=df["item_name"],
            mode="markers",
            name="Reorder Mark",
            marker=dict(symbol="x", size=10, color=INK, line=dict(color=INK, width=2.5)),
            hovertemplate="Reorder mark for <b>%{y}</b>: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        **make_doodle_layout(
            f"Nearest Reorder Marks · {len(df)} of {len(inventory_df)} SKUs",
            x_title="Units",
        ),
        height=max(360, 90 + len(df) * 28),
    )
    return fig


def build_doodle_forecast_chart(
    forecast_df: pd.DataFrame,
    sku_label: str,
    sales_history: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    if sales_history is not None and not sales_history.empty:
        history = sales_history.copy()
        history["sale_date"] = pd.to_datetime(history["sale_date"], errors="coerce")
        daily_history = (
            history.dropna(subset=["sale_date"])
            .groupby("sale_date", as_index=False)
            .agg(units=("quantity", "sum"))
            .sort_values("sale_date")
            .tail(14)
        )
        if not daily_history.empty:
            fig.add_trace(
                go.Scatter(
                    x=daily_history["sale_date"],
                    y=daily_history["units"],
                    mode="lines+markers",
                    name="Recorded Units",
                    line=dict(color=TEAL, width=3),
                    marker=dict(size=7, color=TEAL, line=dict(color=INK, width=1.5)),
                    hovertemplate=(
                        "<b>%{x|%b %d}</b><br>"
                        "Recorded: %{y:,.0f} units<extra></extra>"
                    ),
                )
            )
    fig.add_trace(
        go.Bar(
            x=forecast_df["forecast_date"],
            y=forecast_df["predicted_units"],
            name="Forecast Units",
            marker=dict(color=MUSTARD, line=dict(color=INK, width=2)),
            customdata=forecast_df["predicted_revenue"].map(format_inr),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Forecast: %{y:.1f} units<br>"
                "Est. Sales: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["forecast_date"],
            y=forecast_df["predicted_units"],
            mode="lines+markers",
            name="Forecast Trend",
            line=dict(color=CORAL, width=2.5, shape="linear"),
            marker=dict(size=7, color=INK),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **make_doodle_layout(
            f"Recorded vs Forecast · {sku_label}",
            y_title="Units Sold / Predicted",
        )
    )
    return fig


def build_advisor_chart(
    prompt: str,
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> Optional[go.Figure]:
    chart_type = requested_chart_type(prompt)
    currency_values = None

    if chart_type == "profit":
        profit = summarize_sku_profit(sales_df).dropna(subset=["gross_profit"])
        if profit.empty:
            return None
        plot_data = profit.nlargest(14, "gross_profit").sort_values("gross_profit")
        fig = go.Figure(
            go.Bar(
                x=plot_data["gross_profit"],
                y=plot_data["item_name"],
                orientation="h",
                marker=dict(
                    color=[
                        TEAL if value >= 0 else CORAL
                        for value in plot_data["gross_profit"]
                    ],
                    line=dict(color=INK, width=2),
                ),
                customdata=np.column_stack(
                    [plot_data["sku"], plot_data["gross_profit"].map(format_inr)]
                ),
                hovertemplate=(
                    "<b>%{y}</b> (%{customdata[0]})<br>"
                    "Gross profit: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        title = "Gross Profit Doodle · Top 14 SKUs"
        x_title = "Gross Profit (₹)"
        plot_count = len(plot_data)
        currency_values = plot_data["gross_profit"]
    elif chart_type == "forecast":
        return (
            build_doodle_forecast_chart(forecast_df, "All SKUs", sales_df)
            if not forecast_df.empty
            else None
        )
    elif chart_type == "inventory":
        return build_doodle_stock_chart(inventory_df) if not inventory_df.empty else None
    elif chart_type == "sku_sales":
        if sales_df.empty:
            return None
        sku_sales = (
            sales_df.groupby(["sku", "item_name"], as_index=False)
            .agg(revenue=("total_amount", "sum"), units=("quantity", "sum"))
            .nlargest(14, "revenue")
            .sort_values("revenue")
        )
        fig = go.Figure(
            go.Bar(
                x=sku_sales["revenue"],
                y=sku_sales["item_name"],
                orientation="h",
                marker=dict(color=MUSTARD, line=dict(color=INK, width=2)),
                customdata=np.column_stack(
                    [
                        sku_sales["sku"],
                        sku_sales["units"],
                        sku_sales["revenue"].map(format_inr),
                    ]
                ),
                hovertemplate=(
                    "<b>%{y}</b> (%{customdata[0]})<br>"
                    "Revenue: %{customdata[2]}<br>"
                    "Units: %{customdata[1]}<extra></extra>"
                ),
            )
        )
        title = "Best-Selling Shelf Notes · Top 14 SKUs"
        x_title = "Sales (₹)"
        plot_count = len(sku_sales)
        currency_values = sku_sales["revenue"]
    elif chart_type == "weekly_sales":
        return build_doodle_gross_sales_chart(sales_df) if not sales_df.empty else None
    else:
        return None

    fig.update_layout(
        **make_doodle_layout(title, x_title=x_title),
        height=max(360, 90 + plot_count * 28),
    )
    if currency_values is not None:
        fig.update_xaxes(**indian_currency_tick_options(currency_values))
    return fig


def compute_demand_forecast(
    sales_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sku_filter: str = "ALL",
    horizon_days: int = 7,
) -> pd.DataFrame:
    if sales_df.empty:
        return pd.DataFrame()

    df = sales_df.copy()
    if sku_filter != "ALL":
        df = df[df["sku"] == sku_filter]

    if df.empty:
        return pd.DataFrame()

    df["sale_dt"] = pd.to_datetime(df["sale_date"], errors="coerce")
    df = df.dropna(subset=["sale_dt"])
    daily = df.groupby("sale_dt", as_index=False).agg(
        units=("quantity", "sum"),
        revenue=("total_amount", "sum"),
    )

    if len(daily) < 2:
        return pd.DataFrame()

    start_dt = daily["sale_dt"].min()
    all_dates = pd.date_range(start_dt, daily["sale_dt"].max(), freq="D")
    daily = (
        daily.set_index("sale_dt")
        .reindex(all_dates, fill_value=0)
        .rename_axis("sale_dt")
        .reset_index()
    )
    units = daily["units"].to_numpy(dtype=float)

    model = None
    if len(units) >= 14:
        training_rows = []
        for day_index in range(7, len(units)):
            trailing_units = units[max(0, day_index - 28) : day_index]
            training_rows.append(
                [
                    float(day_index),
                    units[day_index - 1],
                    units[day_index - 7],
                    float(np.mean(units[day_index - 7 : day_index])),
                    float(np.mean(trailing_units)),
                    float((start_dt + timedelta(days=day_index)).weekday() >= 5),
                ]
            )
        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(np.asarray(training_rows), units[7:])

    avg_unit_price = (
        float(df["total_amount"].sum() / df["quantity"].sum())
        if df["quantity"].sum() > 0
        else 0.0
    )
    if sku_filter != "ALL" and not inventory_df.empty:
        match = inventory_df[inventory_df["sku"] == sku_filter]
        if not match.empty:
            avg_unit_price = float(match.iloc[0]["unit_price"])

    last_dt = daily["sale_dt"].max()
    forecast_history = units.tolist()
    short_history_average = float(np.mean(units[-7:]))
    rows = []
    for step in range(1, horizon_days + 1):
        f_dt = last_dt + timedelta(days=step)
        if model is None:
            pred_units = short_history_average
        else:
            day_index = len(forecast_history)
            trailing_units = forecast_history[max(0, day_index - 28) : day_index]
            x_pred = np.asarray(
                [
                    [
                        float(day_index),
                        forecast_history[-1],
                        forecast_history[-7],
                        float(np.mean(forecast_history[-7:])),
                        float(np.mean(trailing_units)),
                        float(f_dt.weekday() >= 5),
                    ]
                ]
            )
            pred_units = max(0.0, float(model.predict(x_pred)[0]))
        forecast_history.append(pred_units)
        rows.append(
            {
                "forecast_date": f_dt.strftime("%Y-%m-%d"),
                "predicted_units": round(pred_units, 2),
                "predicted_revenue": round(pred_units * avg_unit_price, 2),
            }
        )

    return pd.DataFrame(rows)


def evaluate_demand_forecast(
    sales_df: pd.DataFrame,
    sku_filter: str = "ALL",
    horizon_days: int = 7,
    max_windows: int = 4,
) -> Dict[str, Any]:
    if sales_df.empty or horizon_days <= 0 or max_windows <= 0:
        return {"ready": False, "reason": "Not enough sales history to backtest."}

    history = sales_df.copy()
    if sku_filter != "ALL":
        history = history[history["sku"] == sku_filter]
    if history.empty:
        return {"ready": False, "reason": "No sales history is available for this SKU."}

    history["sale_dt"] = pd.to_datetime(history["sale_date"], errors="coerce").dt.normalize()
    history = history.dropna(subset=["sale_dt"])
    daily = history.groupby("sale_dt", as_index=False).agg(units=("quantity", "sum"))
    if daily.empty:
        return {"ready": False, "reason": "No valid dated sales are available for backtesting."}

    dates = pd.date_range(daily["sale_dt"].min(), daily["sale_dt"].max(), freq="D")
    daily_units = (
        daily.set_index("sale_dt")["units"]
        .reindex(dates, fill_value=0)
        .to_numpy(dtype=float)
    )
    window_count = min(max_windows, (len(daily_units) - 14) // horizon_days)
    if window_count < 1:
        return {
            "ready": False,
            "reason": (
                f"Backtesting needs at least {14 + horizon_days} days of history "
                f"for a {horizon_days}-day evaluation window."
            ),
        }

    model_actuals = []
    model_predictions = []
    baseline_predictions = []
    first_test_index = len(daily_units) - window_count * horizon_days

    for window in range(window_count):
        test_start = first_test_index + window * horizon_days
        test_end = test_start + horizon_days
        cutoff_date = dates[test_start - 1]
        training_sales = history[history["sale_dt"] <= cutoff_date].drop(columns="sale_dt").copy()
        if not (training_sales["sale_date"].map(pd.Timestamp) == cutoff_date).any():
            anchor = {column: None for column in training_sales.columns}
            anchor.update(
                {
                    "sale_date": cutoff_date.strftime("%Y-%m-%d"),
                    "quantity": 0,
                    "total_amount": 0.0,
                }
            )
            training_sales = pd.concat(
                [training_sales, pd.DataFrame([anchor])],
                ignore_index=True,
            )
        forecasts = compute_demand_forecast(
            sales_df=training_sales,
            inventory_df=pd.DataFrame(),
            horizon_days=horizon_days,
        )
        if len(forecasts) != horizon_days:
            continue

        model_actuals.extend(daily_units[test_start:test_end])
        model_predictions.extend(forecasts["predicted_units"].to_numpy(dtype=float))

        window_baseline = []
        for offset in range(horizon_days):
            if offset < 7:
                prediction = daily_units[test_start + offset - 7]
            else:
                prediction = window_baseline[offset - 7]
            window_baseline.append(float(prediction))
        baseline_predictions.extend(window_baseline)

    if not model_actuals:
        return {"ready": False, "reason": "Could not produce complete backtest windows."}

    actuals = np.asarray(model_actuals, dtype=float)
    predictions = np.asarray(model_predictions, dtype=float)
    baseline = np.asarray(baseline_predictions, dtype=float)
    model_errors = np.abs(actuals - predictions)
    baseline_errors = np.abs(actuals - baseline)
    actual_total = float(actuals.sum())

    return {
        "ready": True,
        "windows": len(model_actuals) // horizon_days,
        "evaluated_days": len(model_actuals),
        "horizon_days": horizon_days,
        "model_mae": float(model_errors.mean()),
        "model_wape": float(model_errors.sum() / actual_total * 100) if actual_total else None,
        "baseline_mae": float(baseline_errors.mean()),
        "baseline_wape": float(baseline_errors.sum() / actual_total * 100) if actual_total else None,
    }


def main() -> None:
    st.set_page_config(
        page_title="Dukaan Mind",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_db()
    seed_sample_data()
    inject_doodle_notebook_css()

    if "show_api_vault" not in st.session_state:
        st.session_state["show_api_vault"] = False

    settings = get_settings()
    shop_name = settings.get("shop_name") or os.environ.get("DUKAAN_SHOP_NAME", "My Dukaan")

    with st.container(border=True, key="doodle-header"):
        st.markdown(
            """
            <div class="doodle-toprow">
                <span>dukaan mind · counter notebook</span>
                <span>SQLite · Ridge · Shop Advisor</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        h_left, h_right = st.columns([3, 1.5], vertical_alignment="center")
        with h_left:
            st.markdown(
                f"""
                <h1 class="doodle-title">{escape_html(str(shop_name))}</h1>
                <div class="doodle-sub">your shop at a glance · sales · stock · forecasts</div>
                """,
                unsafe_allow_html=True,
            )
        with h_right:
            if st.button(
                "Shop Advisor Settings",
                key="hidden_doodle_api",
                help="Configure the Shop Advisor and shop details",
                width="stretch",
            ):
                st.session_state["show_api_vault"] = not st.session_state["show_api_vault"]
                st.rerun()

    if st.session_state["show_api_vault"]:
        with st.expander("Shop Advisor Settings", expanded=True):
            with st.form("hidden_doodle_setup_form"):
                c1, c2 = st.columns(2)
                api_key_in = c1.text_input(
                    "Shop Advisor API Key",
                    value=settings.get("gemini_api_key", ""),
                    type="password",
                )
                model_options = [
                    "gemini-3.5-flash-lite",
                    "gemini-3.6-flash",
                    "gemini-3.8-flash",
                    "gemini-4-hydrogen",
                ]
                cur_model = settings.get("gemini_model", "gemini-3.5-flash-lite")
                m_idx = model_options.index(cur_model) if cur_model in model_options else 0
                model_in = c2.selectbox("Model", options=model_options, index=m_idx)

                c3, c4 = st.columns(2)
                shop_in = c3.text_input("Shop Name", value=shop_name)
                rules_in = c4.text_input(
                    "Shop Notes / Rules",
                    value=settings.get("shop_instructions", ""),
                )

                b1, b2 = st.columns(2)
                if b1.form_submit_button("Save Notes", width="stretch"):
                    save_settings(
                        gemini_api_key=api_key_in,
                        gemini_model=model_in,
                        shop_name=shop_in,
                        shop_instructions=rules_in,
                    )
                    st.success("Saved.")
                    st.rerun()
                if b2.form_submit_button("Tuck Away", width="stretch"):
                    st.session_state["show_api_vault"] = False
                    st.rerun()

    settings = get_settings()
    inventory_df = get_inventory_df()
    sales_df = get_sales_df()

    low_stock_df = (
        inventory_df[inventory_df["stock_qty"] <= inventory_df["reorder_level"]]
        if not inventory_df.empty
        else pd.DataFrame()
    )
    low_stock_count = len(low_stock_df)

    if low_stock_count > 0:
        flagged = ", ".join(
            f"{r['item_name']} ({int(r['stock_qty'])}/{int(r['reorder_level'])})"
            for _, r in low_stock_df.iterrows()
        )
        st.markdown(
            f'<div class="note-banner urgent">Circle these to restock ({low_stock_count}): {flagged}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="note-banner">All shelf items look healthy above their reorder marks.</div>',
            unsafe_allow_html=True,
        )

    total_revenue = float(sales_df["total_amount"].sum()) if not sales_df.empty else 0.0
    total_units = int(sales_df["quantity"].sum()) if not sales_df.empty else 0
    sku_count = len(inventory_df)
    profit_summary = summarize_sku_profit(sales_df)
    gross_profit = (
        float(profit_summary["gross_profit"].fillna(0).sum())
        if not profit_summary.empty
        else 0.0
    )
    costed_units = (
        int(profit_summary["costed_units"].sum())
        if not profit_summary.empty
        else 0
    )

    if sales_df.empty:
        top_seller_card = (
            "Top mover",
            "No sales yet",
            "Log a few sales to see which item moves fastest.",
        )
    else:
        sales_by_sku = (
            sales_df.groupby(["sku", "item_name"], as_index=False)
            .agg(units_sold=("quantity", "sum"), revenue=("total_amount", "sum"))
            .sort_values(["units_sold", "revenue"], ascending=[False, False])
        )
        top_seller = sales_by_sku.iloc[0]
        top_seller_card = (
            "Top mover",
            str(top_seller["item_name"]),
            f"{format_indian_number(int(top_seller['units_sold']))} units sold · "
            f"{format_inr(top_seller['revenue'])} revenue across recorded sales",
        )

    if low_stock_df.empty:
        restock_card = (
            "Restock focus",
            "Shelf looks healthy",
            "Every item is above its reorder mark.",
        )
    else:
        restock_items = (
            low_stock_df.assign(
                stock_gap=low_stock_df["stock_qty"] - low_stock_df["reorder_level"]
            )
            .sort_values("stock_gap")
            .head(2)
        )
        restock_details = " · ".join(
            f"{row['item_name']} ({format_indian_number(int(row['stock_qty']))}/"
            f"{format_indian_number(int(row['reorder_level']))})"
            for _, row in restock_items.iterrows()
        )
        restock_card = (
            "Restock focus",
            f"{format_indian_number(low_stock_count)} items to check",
            restock_details,
        )

    known_profits = profit_summary.dropna(subset=["gross_profit"])
    if known_profits.empty:
        profit_leader_card = (
            "Profit leader",
            "Cost data needed",
            "Add purchase costs to compare gross profit by item.",
        )
    else:
        profit_leader = known_profits.nlargest(1, "gross_profit").iloc[0]
        profit_leader_card = (
            "Profit leader",
            str(profit_leader["item_name"]),
            f"{format_inr(profit_leader['gross_profit'])} gross profit · "
            f"{format_indian_number(int(profit_leader['costed_units']))} costed units",
        )

    render_sketch_cards(
        [
            ("Total Revenue", format_inr(total_revenue), "counter cashflow", False),
            ("Units Sold", format_indian_number(total_units), "items gone off shelf", False),
            ("Active SKUs", str(sku_count), "listed in notebook", False),
            (
                "Low Stock",
                str(low_stock_count),
                "needs restock note" if low_stock_count else "all clear",
                low_stock_count > 0,
            ),
            (
                "Gross Profit",
                format_inr(gross_profit),
                f"known cost on {format_indian_number(costed_units)} units",
                False,
            ),
        ]
    )

    render_squiggle_rule()

    tab_dash, tab_ops, tab_forecast, tab_ai = st.tabs(
        ["Overview", "Sales & Inventory", "Forecast", "Advisor"]
    )

    with tab_dash:
        st.subheader("Shopkeeper's Quick Read")
        st.caption("Highlights from all recorded sales and current shelf levels.")
        render_insight_cards(
            [restock_card, top_seller_card, profit_leader_card]
        )

        st.subheader("Money & Shelf Sketches")
        st.caption(
            "Sales and gross profit are sketched week by week. Profit excludes sales with no saved purchase cost; "
            "the shelf sketch highlights the 14 SKUs nearest their reorder marks."
        )
        sales_col, profit_col = st.columns(2)
        with sales_col:
            if not sales_df.empty:
                st.plotly_chart(
                    build_doodle_gross_sales_chart(sales_df),
                    width="stretch",
                    key="overview-gross-sales-chart",
                )
            else:
                st.info("No sales logged yet.")
        with profit_col:
            profit_chart = build_doodle_gross_profit_chart(sales_df)
            if profit_chart is not None:
                st.plotly_chart(
                    profit_chart,
                    width="stretch",
                    key="overview-gross-profit-chart",
                )
            else:
                st.info("Enter purchase costs when logging items to sketch gross profit.")

        render_squiggle_rule()
        st.subheader("Shelf Watch")
        if not inventory_df.empty:
            st.plotly_chart(
                build_doodle_stock_chart(inventory_df),
                width="stretch",
                key="overview-stock-chart",
            )
        else:
            st.info("No shelf items yet.")

        render_squiggle_rule()
        st.subheader("Shelf Inventory List")
        inventory_display = inventory_df[
            [
                "sku",
                "item_name",
                "category",
                "stock_qty",
                "reorder_level",
                "unit_price",
                "purchase_cost",
            ]
        ].copy()
        for column in ("unit_price", "purchase_cost"):
            if column in inventory_display:
                inventory_display[column] = inventory_display[column].map(
                    lambda value: format_inr(value) if pd.notna(value) else "—"
                )
        for column in ("stock_qty", "reorder_level"):
            inventory_display[column] = inventory_display[column].map(format_indian_number)
        inventory_display = inventory_display.rename(
            columns={
                "sku": "SKU",
                "item_name": "Item",
                "category": "Category",
                "stock_qty": "On Hand",
                "reorder_level": "Reorder At",
                "unit_price": "Price (₹)",
                "purchase_cost": "Cost (₹)",
            }
        )
        st.dataframe(inventory_display, width="stretch", hide_index=True)

    with tab_ops:
        st.caption("Log a counter sale, update shelf items, or review recent activity here.")
        with st.expander("Indian Grocery Demo Data", expanded=False):
            st.caption(
                "Add a broad kirana catalog and 180 days of realistic sample sales. "
                "Existing items and sales are never overwritten."
            )
            if st.button("Add Demo Groceries", key="add_indian_grocery_demo"):
                added_skus, added_sales = seed_indian_grocery_demo_data()
                if added_skus:
                    st.session_state["demo_data_message"] = (
                        f"Added {format_indian_number(added_skus)} grocery SKUs and "
                        f"{format_indian_number(added_sales)} sample sales."
                    )
                    st.rerun()
                else:
                    st.info("The demo dataset is already loaded; no duplicates were added.")

        if "demo_data_message" in st.session_state:
            st.success(st.session_state.pop("demo_data_message"))

        st.subheader("Jot Down a Sale")
        if not inventory_df.empty:
            sku_choices = inventory_df["sku"].tolist()
            with st.form("record_sale_form"):
                s1, s2, s3 = st.columns(3)
                sale_dt = s1.date_input("Sale Date", value=date.today())
                selected_sku = s2.selectbox("SKU", options=sku_choices)
                sale_qty = s3.number_input("Quantity", min_value=1, value=1, step=1)
                if st.form_submit_button("Log Sale", width="stretch"):
                    err = record_sale(sale_dt.strftime("%Y-%m-%d"), selected_sku, int(sale_qty))
                    if err:
                        st.error(err)
                    else:
                        st.success("Sale logged and stock updated.")
                        st.rerun()

        render_squiggle_rule()
        st.subheader("Add or Edit Shelf Item")
        with st.form("upsert_item_form"):
            i1, i2, i3 = st.columns(3)
            sku_in = i1.text_input("SKU", placeholder="MILK-1L")
            name_in = i2.text_input("Item Name", placeholder="Toned Milk 1L")
            cat_in = i3.text_input("Category", value="Dairy")

            i4, i5, i6, i7 = st.columns(4)
            stock_in = i4.number_input("Stock Qty", min_value=0, value=20, step=1)
            reorder_in = i5.number_input("Reorder Level", min_value=0, value=5, step=1)
            price_in = i6.number_input(
                "Selling Price / Unit (₹)",
                min_value=0.0,
                value=60.0,
                step=1.0,
            )
            cost_known = i7.checkbox("Enter purchase cost")
            purchase_cost_in = i7.number_input(
                "Purchase Cost / Unit (₹)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not cost_known,
            )

            if st.form_submit_button("Save Item", width="stretch"):
                if not sku_in.strip() or not name_in.strip():
                    st.error("SKU and Item Name are required.")
                else:
                    upsert_inventory_item(
                        sku=sku_in,
                        item_name=name_in,
                        category=cat_in,
                        stock_qty=int(stock_in),
                        reorder_level=int(reorder_in),
                        unit_price=float(price_in),
                        purchase_cost=float(purchase_cost_in) if cost_known else None,
                    )
                    st.success("Item saved to notebook.")
                    st.rerun()

        render_squiggle_rule()
        st.subheader("Recent Counter Entries")
        sales_display = sales_df.tail(25)[
            ["sale_date", "sku", "item_name", "quantity", "total_amount"]
        ].copy()
        sales_display["quantity"] = sales_display["quantity"].map(format_indian_number)
        sales_display["total_amount"] = sales_display["total_amount"].map(format_inr)
        sales_display = sales_display.rename(
            columns={
                "sale_date": "Date",
                "sku": "SKU",
                "item_name": "Item",
                "quantity": "Qty",
                "total_amount": "Total (₹)",
            }
        )
        st.dataframe(sales_display, width="stretch", hide_index=True)

        render_squiggle_rule()
        st.subheader("Gross Profit by SKU")
        st.caption(
            "Profit uses purchase cost captured when each sale was logged. "
            "Older sales without cost stay uncalculated."
        )
        sku_profit = summarize_sku_profit(sales_df)
        if sku_profit.empty:
            st.info("Log a sale with a known purchase cost to start the profit notes.")
        else:
            profit_display = sku_profit[
                ["sku", "item_name", "units_sold", "costed_units", "gross_profit", "uncosted_units"]
            ].rename(
                columns={
                    "sku": "SKU",
                    "item_name": "Item",
                    "units_sold": "Units Sold",
                    "costed_units": "Units With Cost",
                    "gross_profit": "Gross Profit (₹)",
                    "uncosted_units": "Units Without Cost",
                }
            )
            for column in ("Units Sold", "Units With Cost", "Units Without Cost"):
                profit_display[column] = profit_display[column].map(format_indian_number)
            profit_display["Gross Profit (₹)"] = profit_display["Gross Profit (₹)"].map(
                lambda value: format_inr(value) if pd.notna(value) else "Unavailable"
            )
            st.dataframe(profit_display, width="stretch", hide_index=True)

    with tab_forecast:
        st.subheader("Recorded Sales & Demand Forecast")
        st.caption(
            "Gold shows the daily Ridge baseline from recent demand and rolling 7-/28-day levels. "
            "The recurring weekly shape reflects the historical weekend effect. Daily values are "
            "estimates, so actual sales can vary. Missing days count as zero; short histories use the "
            "recent daily average."
        )
        f1, f2 = st.columns(2)
        sku_options = ["ALL"] + (inventory_df["sku"].tolist() if not inventory_df.empty else [])
        selected_forecast_sku = f1.selectbox("Target SKU", options=sku_options)
        horizon = f2.slider("Days Ahead", min_value=3, max_value=30, value=7)

        forecast_df = compute_demand_forecast(
            sales_df=sales_df,
            inventory_df=inventory_df,
            sku_filter=selected_forecast_sku,
            horizon_days=horizon,
        )

        if forecast_df.empty:
            st.info("Add at least two distinct days of sales data to sketch a forecast.")
        else:
            history_for_chart = (
                sales_df
                if selected_forecast_sku == "ALL"
                else sales_df[sales_df["sku"] == selected_forecast_sku]
            )
            st.plotly_chart(
                build_doodle_forecast_chart(
                    forecast_df,
                    selected_forecast_sku,
                    history_for_chart,
                ),
                width="stretch",
                key="forecast-chart",
            )
            backtest = evaluate_demand_forecast(
                sales_df=sales_df,
                sku_filter=selected_forecast_sku,
                horizon_days=horizon,
            )
            st.subheader("Forecast Accuracy Check")
            if backtest["ready"]:
                ridge_wape = backtest["model_wape"]
                baseline_wape = backtest["baseline_wape"]
                wape_delta = (
                    f"{baseline_wape - ridge_wape:+.1f} percentage points vs baseline"
                    if ridge_wape is not None and baseline_wape is not None
                    else None
                )
                accuracy_cols = st.columns(3)
                accuracy_cols[0].metric(
                    "Ridge WAPE",
                    f"{ridge_wape:.1f}%" if ridge_wape is not None else "N/A",
                    delta=wape_delta,
                )
                accuracy_cols[1].metric(
                    "Same weekday last week",
                    f"{baseline_wape:.1f}%" if baseline_wape is not None else "N/A",
                )
                accuracy_cols[2].metric(
                    "Ridge MAE",
                    f"{backtest['model_mae']:.1f} units",
                )
                st.caption(
                    f"Walk-forward evaluation across {backtest['windows']} non-overlapping "
                    f"{horizon}-day windows ({backtest['evaluated_days']} held-out days). "
                    "WAPE and MAE are calculated on dates the forecast did not train on; lower is better."
                )
            else:
                st.info(backtest["reason"])

            forecast_display = forecast_df[
                ["forecast_date", "predicted_units", "predicted_revenue"]
            ].copy()
            forecast_display["predicted_units"] = forecast_display[
                "predicted_units"
            ].map(lambda value: f"{value:.1f}")
            forecast_display["predicted_revenue"] = forecast_display[
                "predicted_revenue"
            ].map(format_inr)
            forecast_display = forecast_display.rename(
                columns={
                    "forecast_date": "Date",
                    "predicted_units": "Predicted Units",
                    "predicted_revenue": "Est. Revenue (₹)",
                }
            )
            st.dataframe(forecast_display, width="stretch", hide_index=True)

    with tab_ai:
        st.subheader("Ask the Shop Advisor")
        st.caption(
            "Ask about stock, sales, forecasts, or profit. Add “graph” or “chart” to get a quick sketch from your shop data."
        )
        forecast_for_ai = compute_demand_forecast(
            sales_df=sales_df,
            inventory_df=inventory_df,
            sku_filter="ALL",
            horizon_days=7,
        )
        user_prompt = st.text_area(
            "What would you like to explore?",
            height=105,
            placeholder="Graph weekly sales, compare revenue by SKU, or show my best gross-profit items.",
        )
        if st.button("Ask Advisor", width="stretch"):
            if not user_prompt.strip():
                st.warning("Type a question first.")
            else:
                try:
                    if requests_chart(user_prompt):
                        advisor_chart = build_advisor_chart(
                            user_prompt,
                            inventory_df,
                            sales_df,
                            forecast_for_ai,
                        )
                        if advisor_chart is None:
                            st.info(
                                "There isn’t enough recorded data for that graph yet. "
                                "Log some sales or enter purchase costs for profit graphs."
                            )
                        else:
                            st.caption("A quick doodle, drawn from your recorded shop data.")
                            st.plotly_chart(
                                advisor_chart,
                                width="stretch",
                                key="advisor-chart",
                            )

                    advice_stream = generate_shop_advice_stream(
                        prompt=user_prompt,
                        settings=settings,
                        inventory_df=inventory_df,
                        sales_df=sales_df,
                        forecast_df=forecast_for_ai,
                    )
                    started_at = time.monotonic()

                    def elapsed_label(message):
                        elapsed = int(time.monotonic() - started_at)
                        minutes, seconds = divmod(elapsed, 60)
                        return f"{message} · {minutes}:{seconds:02d}"

                    with st.status(
                        elapsed_label("Sending your question and useful shop details..."),
                        expanded=True,
                    ) as progress:
                        def stream_with_progress():
                            response_queue = queue.Queue()

                            def generate_response():
                                try:
                                    for text in advice_stream:
                                        response_queue.put(("text", text))
                                except Exception as exc:
                                    response_queue.put(("error", exc))
                                finally:
                                    response_queue.put(("done", None))

                            worker = threading.Thread(
                                target=generate_response,
                                daemon=True,
                            )
                            worker.start()
                            received_response = False
                            last_status_update = started_at

                            while True:
                                try:
                                    event, value = response_queue.get(timeout=1)
                                except queue.Empty:
                                    label = (
                                        "The Shop Advisor is putting your answer together..."
                                        if received_response
                                        else "Sending your question and useful shop details..."
                                    )
                                    progress.update(
                                        label=elapsed_label(label),
                                        state="running",
                                    )
                                    last_status_update = time.monotonic()
                                    continue

                                if event == "text":
                                    if not received_response:
                                        received_response = True
                                        progress.update(
                                            label=elapsed_label(
                                                "The Shop Advisor is putting your answer together..."
                                            ),
                                            state="running",
                                        )
                                        last_status_update = time.monotonic()
                                    if time.monotonic() - last_status_update >= 1:
                                        progress.update(
                                            label=elapsed_label(
                                                "The Shop Advisor is putting your answer together..."
                                            ),
                                            state="running",
                                        )
                                        last_status_update = time.monotonic()
                                    yield value
                                elif event == "error":
                                    progress.update(
                                        label=elapsed_label("The advisor hit a snag."),
                                        state="error",
                                    )
                                    raise value
                                else:
                                    progress.update(
                                        label=elapsed_label("Advice is ready!"),
                                        state="complete",
                                    )
                                    return

                        with st.container(border=True):
                            st.write_stream(stream_with_progress())
                except TimeoutError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Request failed: {exc}")


if __name__ == "__main__":
    main()