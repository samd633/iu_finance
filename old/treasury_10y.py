"""
10-Year US Treasury yield over the last 10 years via FMP API.
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


def fetch_treasury() -> pd.DataFrame:
    today = date.today()
    ten_years_ago = date(today.year - 10, today.month, today.day)

    resp = requests.get(
        f"{FMP_BASE}/treasury-rates",
        params={
            "from": ten_years_ago.strftime("%Y-%m-%d"),
            "to": today.strftime("%Y-%m-%d"),
            "apikey": API_KEY,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError("No treasury rate data returned from FMP API.")

    df = pd.DataFrame(data)[["date", "year10"]].copy()
    df = df.rename(columns={"year10": "rate"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["rate"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["rate"],
        mode="lines",
        line=dict(color="#2c7bb6", width=1.5),
        name="10-Year Treasury Yield",
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="10-Year US Treasury Yield — Last 10 Years",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Date",
            tickformat="%Y",
            dtick="M12",
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        yaxis=dict(
            title="Yield (%)",
            ticksuffix="%",
            showgrid=True,
            gridcolor="#e5e5e5",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


if __name__ == "__main__":
    print("Fetching 10-Year Treasury data from FMP...")
    df = fetch_treasury()
    print(f"  Retrieved {len(df)} daily observations.")

    fig = build_chart(df)
    fig.show()
