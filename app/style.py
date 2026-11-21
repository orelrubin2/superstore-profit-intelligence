"""Shared colors, card styling and number formatting for the dashboard (dark theme).

The app runs in a dark theme (.streamlit/config.toml). The colors here are tuned
for that navy background: bright, high-contrast series colors (each at least 5.5:1
against the card background) and light grey text and gridlines. The notebooks keep
their own light palette for the PNG figures in reports/figures/.
"""

import streamlit as st

# ---- surfaces (keep in sync with .streamlit/config.toml)
BACKGROUND = "#0e1729"   # page background (dark navy)
CARD = "#16223a"         # card background, slightly lighter than the page
BORDER = "#33435f"       # thin card border; also outlines of map states with no orders
TEXT = "#e6e9ef"         # light grey text, used for data labels
MUTED = "#9aa6bb"        # secondary text, reference lines, de-emphasised bars
GRID = "#2b3a57"         # gridlines and zero rules

# ---- series colors
BLUE = "#5aa9ff"      # profit / categorical slot 1
ORANGE = "#ff9a52"    # categorical slot 2
GREEN = "#3ddc97"     # categorical slot 3
YELLOW = "#f5c542"    # categorical slot 4
LOSS = "#ff6b6b"      # loss pole of the blue <-> red diverging pair
REGION_COLORS = {"Central": BLUE, "East": ORANGE, "South": GREEN, "West": YELLOW}
CATEGORY_COLORS = {"Furniture": BLUE, "Office Supplies": ORANGE, "Technology": GREEN}
SEQUENTIAL = ["#24486f", "#a8d8ff"]   # dim -> bright blue, low -> high
DIVERGING = [LOSS, "#4a5670", BLUE]   # loss <- slate 0 -> profit

CARD_CSS = f"""
<style>
[class*="st-key-card-"] {{
    background-color: {CARD};
    border-color: {BORDER};
}}
</style>
"""


# Compact page frame for the EDA page and the Order-line checker, so each fits (or nearly
# fits) one 1920x1080 screen: a slim transparent header, less padding, a smaller title,
# tighter spacing and smaller metric values. No font goes below 11 px.
PAGE_CSS = """
<style>
[data-testid="stHeader"] { height: 2rem; background: transparent; }
[data-testid="stMainBlockContainer"] { padding: 2.2rem 2rem 1rem 2rem; }
[data-testid="stMainBlockContainer"] h1 { font-size: 1.7rem; padding: 0 0 1rem 0; }
/* headings keep their 1rem bottom padding: Streamlit offsets it with a negative margin */
[data-testid="stMainBlockContainer"] h3 { font-size: 1.15rem; padding: 0.2rem 0 1rem 0; }
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
[data-testid="stMetricLabel"] { min-height: 0; }
[data-testid="stMetric"] > div:first-child { gap: 0; }
[data-testid="stMetricValue"] { min-height: 0; height: auto; line-height: 1.25; }
[data-testid="stMetricValue"] p { font-size: 1.55rem; }
</style>
"""


def card(key: str):
    """A bordered container with the lighter card background (CSS keyed on the widget key)."""
    return st.container(border=True, key=f"card-{key}")


def money(value: float) -> str:
    """$1,234 or −$1,234 (true minus sign)."""
    return f"−${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def pct(value: float, decimals: int = 1) -> str:
    """12.3% or −12.3%."""
    text = f"{abs(value) * 100:.{decimals}f}%"
    return f"−{text}" if value < 0 else text


def signed_pp(value: float) -> str:
    """+4.2 pp / −4.2 pp for a change in probability."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value) * 100:.1f} pp"
