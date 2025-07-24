import pandas as pd
import requests


def main(url: str, ticker: str) -> None:
    resp = requests.post(url, json={"ticker": ticker}).json()

    # Build DataFrames
    last_day_df = pd.DataFrame.from_records(resp["last_day"])
    forecast_df = pd.DataFrame.from_records(resp["forecast"])

    # Parse dates, set index, and convert returns → percent strings
    for df in (last_day_df, forecast_df):
        # turn the index field back into a datetime index
        df["index"] = pd.to_datetime(df["index"], format="%a, %d %b %Y %H:%M:%S GMT")
        df.set_index("index", inplace=True)

        # convert returns to percent, round to 2 decimal places, and append “%”
        df["returns (%)"] = df["returns"].mul(100).round(2).map(lambda x: f"{x}%")
        df.drop(columns="returns", inplace=True)

    # Print with nicer formatting
    print("\n=== LAST DAY ===")
    print(last_day_df.round({"close": 2, "returns": 6}))
    print("\n=== FORECAST ===")
    print(forecast_df.round({"close": 2, "returns": 6}))


if __name__ == "__main__":
    main(url="http://localhost:9696/forecast", ticker="AMZN")
