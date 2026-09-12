"""
Quarterly US GDP year-over-year growth rate over the last 10 years via FMP API.
"""
import requests
import requests_cache
import pandas as pd
import plotly.graph_objects as go
from datetime import date

FMP_BASE = "https://financialmodelingprep.com/stable"

requests_cache.install_cache("fmp_cache", expire_after=86400, allowable_codes=(200,))

with open("fmp_key.txt") as f:
    API_KEY = f.read().strip()


def fetch_gdp() -> pd.DataFrame:
    today = date.today()
    ten_years_ago = date(today.year - 11, today.month, 1)  # extra year for YoY calc

    resp = requests.get(
        f"{FMP_BASE}/economic-indicators",
        params={
            "name": "GDP",
            "from": ten_years_ago.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "apikey": API_KEY,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No GDP data returned from FMP API.")

    df = pd.DataFrame(data)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"value": "gdp"})
    return df


def build_chart(df: pd.DataFrame) -> go.Figure:
    # Year-over-year growth: compare each quarter to the same quarter a year prior
    df["yoy_growth"] = df["gdp"].pct_change(periods=4) * 100

    cutoff = pd.Timestamp(date.today()) - pd.DateOffset(years=10)
    df = df[df["date"] >= cutoff].dropna(subset=["yoy_growth"])

    colors = ["#d9534f" if v < 0 else "#2c7bb6" for v in df["yoy_growth"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["yoy_growth"],
        marker_color=colors,
        name="YoY GDP Growth (%)",
        hovertemplate="%{x|%Y Q%q}<br>%{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="black", line_width=1)

    fig.update_layout(
        title=dict(
            text="US Quarterly GDP Growth (Year-over-Year) — Last 10 Years",
            font=dict(size=20),
        ),
        xaxis=dict(title="Quarter", tickformat="%Y", dtick="M12"),
        yaxis=dict(title="YoY Growth (%)", ticksuffix="%", zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        bargap=0.2,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


if __name__ == "__main__":
    print("Fetching GDP data from FMP...")
    df = fetch_gdp()
    print(f"  Retrieved {len(df)} quarters total.")

    fig = build_chart(df)
    fig.show()
