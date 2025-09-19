import logging
import sys
import uuid

import pandas as pd
import requests

from gcp_functions import get_gcrun_service_url
from load_configs import Configs
from raw_data_yf import fetch_ticker_data_from_yf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


def main(
    env: str,
    ticker: str,
    past_horizon: int,
    endpoint: str,
    signature_name: str,
    model_id: str,
) -> None:
    # Find the correct URL for the specified environment
    url = find_url_for_env(env, endpoint)

    # Send the request
    if endpoint.split("/")[-1] in ["from_symbol", "from_data"]:
        pl_out = legacy_api_handler(url, endpoint, ticker, past_horizon, model_id)
    elif "/" in endpoint:
        logger.info(f"Sending the ticker symbol to the {endpoint} endpoint")
        if signature_name == "from_symbol":
            pl_in = {
                "ticker": ticker,
                "past_horizon": past_horizon,
                "signature_name": signature_name,
                "model_id": model_id,
            }
        elif signature_name == "from_data":
            try:
                pl_in = build_payload_with_data(
                    ticker=ticker,
                    past_horizon=past_horizon,
                    signature_name=signature_name,
                    model_id=model_id,
                )
            except ValueError as e:
                logger.error(f"Fetching ticker data failed: {e}")
                sys.exit(2)
        try:
            pl_out = post_json(url, pl_in)
        except ApiError as e:
            logger.error(f"API call failed: {e}")
            sys.exit(2)
    else:
        raise ValueError(f"Unrecognized endpoint: {endpoint}")

    meta = pl_out.get("meta", {})
    # Build DataFrames
    past_df = pd.DataFrame.from_records(pl_out["data"]["past"])
    forecast_df = pd.DataFrame.from_records(pl_out["data"]["forecast"])

    # Parse dates, set index, and convert returns → percent strings
    for df in (past_df, forecast_df):
        # turn the index field back into a datetime index
        df["index"] = pd.to_datetime(df["index"], format="%a, %d %b %Y %H:%M:%S GMT")
        df.set_index("index", inplace=True)
        df.index.name = None

        # convert returns to percent, round to 2 decimal places, and append “%”
        if "Returns" in df.columns:
            df["Returns (%)"] = df["Returns"].mul(100).round(2).map(lambda x: f"{x}%")
            df.drop(columns="Returns", inplace=True)

    # Print with nicer formatting
    print("\n=== META ===")
    print(meta)
    print("\n=== PAST PRICES ===")
    print(past_df.round({"Close": 2}))
    print("\n=== FORECAST ===")
    print(forecast_df.round({"Close": 2, "Returns": 6}))


def legacy_api_handler(
    url: str, endpoint: str, ticker: str, past_horizon: int, model_id: str
) -> tuple:
    if endpoint.split("/")[-1] in ["from_symbol"]:  # backward compatibility for API v1
        logger.info(f"Sending the ticker symbol to the {endpoint} endpoint")
        pl_in = {"ticker": ticker, "past_horizon": past_horizon, "model_id": model_id}
    elif endpoint.split("/")[-1] in ["from_data"]:  # backward compatibility for API v1
        logger.info(f"Fetching ticker data and sending it to the {endpoint} endpoint")
        pl_in = build_payload_with_data(
            ticker=ticker, past_horizon=past_horizon, model_id=model_id
        )

    resp = requests.post(url, json=pl_in)
    pl_out = resp.json()

    # This is the new output format of v2
    data = {
        "past": pl_out["past"],
        "forecast": pl_out["forecast"],
    }
    pl_out_new = {"data": data, "meta": pl_out["meta"]}

    return pl_out_new


def find_url_for_env(env: str, endpoint: str) -> str:
    assert env in ["local", "test", "dev", "prod"]

    if env == "local":
        url = f"http://0.0.0.0:9696/{endpoint}"
    else:
        configs = Configs(env).cloud
        service_name_root = configs["gcs"]["service_name_root"]
        url_root = get_gcrun_service_url(
            service_name=f"{service_name_root}-{env}",
            region=configs["gcs"]["region"],
            project_id=configs["gcs"]["project"],
        )
        url = f"{url_root}/{endpoint}"
    print(f"for the requested {env} environment, service url is: {url}")
    return url


def build_payload_with_data(
    ticker: str,
    past_horizon: int,
    signature_name: str = "NA",
    model_id: str | None = None,
) -> dict:
    """
    Fetches data via with fetch_ticker_data_from_yf(ticker)
    and returns the columnar JSON dict expected by the /from_data endpoint.
    """
    df = fetch_ticker_data_from_yf(ticker)

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Date' and 'Close' columns.")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")

    # Clean + order
    df = (
        df.dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
    )

    payload = {
        "ticker": ticker,
        "series": {
            "date": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "close": df["Close"].astype(float).tolist(),
        },
        "past_horizon": past_horizon,
        "signature_name": signature_name,
    }
    if model_id is not None:
        payload["model_id"] = model_id
    return payload


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        title: str,
        detail: str,
        request_id: str | None = None,
        body: dict | None = None,
    ) -> None:
        super().__init__(f"{status} {title}: {detail} (request_id={request_id})")
        self.status = status
        self.title = title
        self.detail = detail
        self.request_id = request_id
        self.body = body


def post_json(
    url: str,
    payload: dict,
    *,
    request_id: str | None = None,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> dict:
    """POST JSON, include/echo X-Request-ID, raise ApiError on non-2xx."""
    rid = request_id or str(uuid.uuid4())
    headers = {
        "Accept": "application/json, application/problem+json",
        "X-Request-ID": rid,
    }
    http = session or requests
    resp = http.post(url, json=payload, headers=headers, timeout=timeout)

    server_rid = resp.headers.get("X-Request-ID", "Unknown")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Try to parse RFC-7807 or generic JSON; fall back to text
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        raise ApiError(
            status=resp.status_code,
            title=body.get("title", "HTTP Error"),
            detail=body.get("detail", ""),
            request_id=server_rid,
            body=body,
        ) from e

    # Success: return JSON
    data = resp.json()
    if isinstance(data, dict):
        meta = data.setdefault("meta", {})
        # if meta doesn't contain request_id, inject server_rid read from the header
        if "request_id" not in meta:
            meta["request_id"] = server_rid
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the forecasting endpoint. Example use: python scripts/test_inference.py --ticker AMZN"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=False,
        help="ticker symbol, e.g., 'AMZN'",
        default="AMZN",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=False,
        help="deployed environment. Options: local, test, dev, prod",
        default="local",
    )
    parser.add_argument(
        "--past_horizon",
        type=int,
        required=False,
        help="number (days) of past prices should be returned",
        default=10,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help="forecasting endpoint; options: forecast, forecast/from_symbol, forecast/from_series",
        # default="v2/forecast"
        default="v1/forecast/from_data",
        # default="v1/forecast/from_symbol",
    )
    parser.add_argument(
        "--signature_name",
        type=str,
        required=False,
        help="signature name, options: from_symbol, from_data",
        default="from_symbol",
        # default="from_data",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=False,
        help="model id, options: an mlflow run id available locally or on gcs",
        default="default",
    )

    args = parser.parse_args()

    main(
        env=args.env,
        ticker=args.ticker,
        past_horizon=args.past_horizon,
        endpoint=args.endpoint,
        signature_name=args.signature_name,
        model_id=args.model_id,
    )
