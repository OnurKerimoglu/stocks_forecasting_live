import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


def call_api(ticker: str, past_horizon: int, env: str, endpoint: str) -> dict:
    if env == "local":
        API_URL = "http://0.0.0.0:9696"
    else:
        API_URL_TEMPLATE = st.secrets["global"]["API_URL_TEMPLATE"]
        API_URL = API_URL_TEMPLATE.replace("ENV", env)
    pl_in = {"ticker": ticker, "past_horizon": past_horizon}
    resp = requests.post(f"{API_URL}/{endpoint}", json=pl_in, timeout=30)
    resp.raise_for_status()
    pl_out = resp.json()
    return pl_out


def build_chart(data: dict, ticker: str) -> go.Figure:
    """Create a Plotly figure with past and forecast 'Close' prices."""
    # Convert to DataFrame & ensure datetime index
    past_df = pd.DataFrame(data["past"]).rename(columns={"index": "Date"})
    fcst_df = pd.DataFrame(data["forecast"]).rename(columns={"index": "Date"})
    # add the last day of past_df to fcst_df
    fcst_df = pd.concat([past_df.iloc[-1:], fcst_df])

    past_df["Date"] = pd.to_datetime(past_df["Date"], utc=True).dt.tz_convert(None)
    fcst_df["Date"] = pd.to_datetime(fcst_df["Date"], utc=True).dt.tz_convert(None)

    # Build figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=past_df["Date"],
            y=past_df["Close"],
            mode="lines+markers",
            name="Past Close",
            line=dict(width=2, color="blue"),
            marker=dict(size=9, symbol="circle", color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fcst_df["Date"],
            y=fcst_df["Close"],
            mode="lines+markers",
            name="Forecast Close",
            line=dict(width=2, dash="dash", color="red"),
            marker=dict(size=9, symbol="circle", color="red"),
        )
    )

    # forecast origin marker
    fig.add_trace(
        go.Scatter(
            x=[fcst_df["Date"].iloc[0]],
            y=[fcst_df["Close"].iloc[0]],
            mode="markers",
            marker=dict(size=11, symbol="diamond", color="violet"),
            name="Forecast origin",
        )
    )

    # Axis formatting
    # Major ticks/labels: every Monday (start of week)
    # Minor ticks: every calendar day
    mondays = pd.date_range(
        start=past_df["Date"].min().normalize(),
        end=fcst_df["Date"].max().normalize(),
        freq="W-MON",
    )
    tickvals = [d.to_pydatetime() for d in mondays]
    ticktext = [d.strftime("%b %d") for d in mondays]

    fig.update_xaxes(
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        minor=dict(
            dtick="D1",
            ticklen=6,
            tickcolor="black",
            tickmode="auto",
            nticks=10,
            showgrid=True,
        ),
    )
    fig.update_yaxes(showgrid=True, zeroline=False)

    fig.update_layout(
        title=f"Closing Price Forecast for: {ticker.upper()}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        font=dict(size=14),
        autosize=False,
        width=1000,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Stock Price Forecast", layout="wide")
    st.title("📈 Stock Price Forecast")

    # --- Sidebar controls ---
    st.sidebar.header("Query Parameters")
    ticker = st.sidebar.text_input("Ticker", value="AMZN", max_chars=10)
    # past_horizon = st.sidebar.slider(
    #     "Past Horizon (business days)",
    #     min_value=0,
    #     max_value=40,
    #     value=20,
    #     step=5,
    # )
    # past_horizon = max(past_horizon, 1)  # ensure at least one day
    past_horizon = 20
    env = st.secrets["global"]["env"]
    endpoint = st.secrets["global"]["endpoint"]

    if st.sidebar.button("Fetch & Plot"):
        with st.spinner("Contacting API…"):
            try:
                st.write(f"Calling API for {env} env")
                payload = call_api(ticker, past_horizon, env, endpoint)
                fig = build_chart(payload, ticker)
                st.plotly_chart(fig, use_container_width=False)
            except requests.HTTPError as e:
                st.error(f"API request failed: {e}")
            except Exception as err:
                st.error(f"❗️ Unexpected error: {err}")


if __name__ == "__main__":
    main()
