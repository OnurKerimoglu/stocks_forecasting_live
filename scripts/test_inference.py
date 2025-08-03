import pandas as pd
import requests
from gcp_functions import get_gcrun_service_url
from load_configs import Configs


def main(env: str, ticker: str, past_horizon: int) -> None:
    # Find the correct URL for the specified environment
    url = find_url_for_env(env)

    # Send the request
    pl_in = {"ticker": ticker, "past_horizon": past_horizon}
    resp = requests.post(url, json=pl_in)
    pl_out = resp.json()

    # Build DataFrames
    past_df = pd.DataFrame.from_records(pl_out["past"])
    forecast_df = pd.DataFrame.from_records(pl_out["forecast"])

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
    print("\n=== PAST PRICES ===")
    print(past_df.round({"Close": 2}))
    print("\n=== FORECAST ===")
    print(forecast_df.round({"Close": 2, "Returns": 6}))


def find_url_for_env(env: str) -> str:
    assert env in ["local", "test", "dev", "prod"]

    if env == "local":
        url = "http://0.0.0.0:9696/forecast"
    else:
        # url = get_static_url_for_env(env)
        configs = Configs(env).cloud
        service_name_root = configs["gcs"]["service_name_root"]
        url_root = get_gcrun_service_url(
            service_name=f"{service_name_root}-{env}",
            region=configs["gcs"]["region"],
            project_id=configs["gcs"]["project"],
        )
        url = f"{url_root}/forecast"
    print(f"for the requested {env} environment, service url is: {url}")
    return url


def get_static_url_for_env(env: str) -> str:
    url = f"https://stocks-forecasting-service-{env}-qlypn5u2fq-ew.a.run.app"
    return url


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the forecasting endpoint. Example use: python scripts/test_inference.py --ticker AMZN"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=False,
        help="ticker symbol, e.g., 'AMZN",
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
        default=0,
    )
    args = parser.parse_args()

    main(env=args.env, ticker=args.ticker, past_horizon=args.past_horizon)
