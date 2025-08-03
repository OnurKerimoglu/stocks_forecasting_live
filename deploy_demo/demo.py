# import google.auth.transport.requests
# import google.oauth2.id_token
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# API_URL = os.environ["API_URL"]
API_URL = "http://0.0.0.0:9696"


def call_api(ticker: str, past_horizon: int) -> dict:
    # Obtain a signed ID token for the audience
    # request = google.auth.transport.requests.Request()
    # token = google.oauth2.id_token.fetch_id_token(request, API_URL)

    # headers = {"Authorization": f"Bearer {token}"}
    pl_in = {"ticker": ticker, "past_horizon": past_horizon}
    resp = requests.post(f"{API_URL}/forecast", json=pl_in, timeout=10)
    resp.raise_for_status()
    pl_out = resp.json()
    return pl_out


def build_chart(data: dict, ticker: str) -> go.Figure:
    """Create a Plotly figure with past and forecast 'Close' prices."""
    # Convert to DataFrame & ensure datetime index
    past_df = pd.DataFrame(data["past"]).rename(columns={"index": "Date"})
    fcst_df = pd.DataFrame(data["forecast"]).rename(columns={"index": "Date"})

    past_df["Date"] = pd.to_datetime(past_df["Date"])  # parses the RFC date string
    fcst_df["Date"] = pd.to_datetime(fcst_df["Date"])

    # Build figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=past_df["Date"],
            y=past_df["Close"],
            mode="lines+markers",
            name="Past Close",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fcst_df["Date"],
            y=fcst_df["Close"],
            mode="lines+markers",
            name="Forecast Close",
            line=dict(width=2, dash="dash"),
        )
    )

    # Adaptive X-axis formatting depending on total horizon
    total_points = len(past_df) + len(fcst_df)
    if total_points <= 15:
        tickformat = "%b %d"  # "Aug 04"
        dtick = None
    elif total_points <= 35:
        tickformat = "%b %d"
        dtick = "M1"  # monthly ticks
    else:
        tickformat = "%b %Y"  # "Aug 2025"
        dtick = "M1"

    fig.update_layout(
        title=f"Closing Price Forecast for {ticker.upper()}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )

    fig.update_xaxes(tickformat=tickformat, dtick=dtick, showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

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
    past_horizon = st.sidebar.slider(
        "Past Horizon (business days)",
        min_value=5,
        max_value=60,
        value=30,
        step=5,
    )

    if st.sidebar.button("Fetch & Plot"):
        with st.spinner("Contacting API…"):
            try:
                payload = call_api(ticker, past_horizon)
                fig = build_chart(payload, ticker)
                st.plotly_chart(fig, use_container_width=True)
            except requests.HTTPError as e:
                st.error(f"API request failed: {e}")
            except Exception as err:
                st.error(f"❗️ Unexpected error: {err}")


if __name__ == "__main__":
    main()
