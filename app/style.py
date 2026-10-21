"""Shared colors and number formatting for the dashboard (same palette as the notebooks)."""

BLUE = "#2a78d6"      # profit / categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
LOSS = "#e34948"      # loss pole of the blue <-> red diverging pair
GREY = "#52514e"      # reference lines and secondary text
BAND_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal ramp, 0% -> >40%


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
