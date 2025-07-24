import datetime
import json
import logging
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

from data import (
    build_features,
    clean_raw_data,
    create_X_y_multistep,
)

# Global parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"

rootpath = os.path.dirname(__file__)
DEPLOYPATH = os.path.join(rootpath, "deployment")

logger = logging.getLogger(__name__)


def stocks_forecasting_inference_flow(
    ticker: str = "AAPL", use_model_registry: bool = False
) -> None:
    if use_model_registry:
        model, params = retrieve_registered_model()
    else:
        model, params = retrieve_locally_stored_model()
    df_init = retrieve_ticker_data(ticker)
    last_day, forecast = run_forecast(model, params, df_init)
    logger.info(f"\nlast day:\n{last_day}\nforecast:\n{forecast}")
    return last_day, forecast


def retrieve_registered_model() -> tuple:
    # Load the model from the Model Registry
    model_uri = f"models:/{REGISTRY_NAME}@{MODEL_ALIAS}"
    logger.info(f"Retrieveing model_uri: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    # Get the parameters
    mv = CLIENT.get_model_version_by_alias(name=REGISTRY_NAME, alias=MODEL_ALIAS)
    run = CLIENT.get_run(mv.run_id)
    params = run.data.params
    return model, params


def retrieve_locally_stored_model() -> tuple:
    fpath = os.path.join(DEPLOYPATH, "model.pkl")
    logger.info(f"Loading model from: {fpath}")
    with open(fpath, "rb") as f:
        model = pickle.load(f)
    fpath = os.path.join(DEPLOYPATH, "params.json")
    logger.info(f"Loading params from: {fpath}")
    with open(fpath) as f:
        params = json.load(f)
    return model, params


def retrieve_ticker_data(ticker: str) -> pd.DataFrame:
    # download the raw data
    df_raw = fetch_ticker_data_from_yf(ticker=ticker)
    # clean the raw data (e.g. winsorize returns)
    df = clean_raw_data(df_raw)
    df.sort_values(["Date"], inplace=True)
    return df


def fetch_ticker_data_from_yf(ticker: str) -> pd.DataFrame:
    period_start = datetime.datetime.now() - datetime.timedelta(days=100)
    logger.info(f"Fetching price data for {ticker}")
    try:
        df = yf.download(
            ticker,
            start=period_start,
            end=datetime.datetime.now(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        raise e from ValueError(f"Failed to fetch data for {ticker}")

    # Get rid of the redundant Ticker index
    df.columns = df.columns.droplevel("Ticker")
    # Re-introduce the ticker as a regular column
    df["Ticker"] = ticker
    # Make 'Date' a regular column by resetting the index
    df.reset_index(inplace=True)

    if df.shape[0] > 0:
        logger.info(f"Fetched {df.shape[0]} rows for {ticker}")
    else:
        raise ValueError(f"No valid raws in fetched data for {ticker}")
    return df


def run_forecast(
    model: XGBRegressor,
    parameters: dict,
    df_init: pd.DataFrame,
    bizday_offset: bool = True,
) -> tuple:
    # Build features on your last observed history
    CldrFeats = parameters["CldrFeats"] if "CldrFeats" in parameters.keys() else True
    df_feats, _ = build_features(
        df_init, lags=int(parameters["lags"]), CldrFeats=CldrFeats
    )
    # create features for the very last day, so specify only 1 step ahead in the future to avoid losing the features
    X_train, y_train = create_X_y_multistep(df_feats, steps=1, target="returns")
    # Grab the last row of features (drop identifiers)
    X_step = X_train.iloc[[-1], :]

    # Predicted returns
    y_hat = model.predict(X_step)[0]

    # Generate the timestamps (business days)
    last_date = df_init["Date"].max()
    if bizday_offset:
        from pandas.tseries.offsets import BDay

        dates_ts = [last_date + BDay(i) for i in range(0, int(parameters["steps"]) + 1)]
    else:
        dates_ts = [
            last_date + pd.Timedelta(days=i)
            for i in range(0, int(parameters["steps"]) + 1)
        ]
    dates = [timestamp.date() for timestamp in dates_ts]
    last_return = y_train.iloc[-1].values[0]
    returns = np.append(last_return, y_hat)
    returns_series = pd.Series(returns, index=dates, name="returns")

    # Compute prices from returns
    # Start from last observed close
    last_close = df_init.loc[df_init["Date"] == last_date, "Close"].iloc[0]
    prices = [last_close]
    price_prev = last_close
    for returns in y_hat:
        price_next = price_prev * (1 + returns)
        prices.append(price_next)
        price_prev = price_next

    prices_series = pd.Series(prices, index=dates, name="close")

    result = pd.concat([returns_series, prices_series], axis=1)

    last = result.iloc[[0], :]
    forecast = result.iloc[1:, :]
    return last, forecast


app = Flask("stocks-forecasting")


@app.route("/forecast", methods=["POST"])
def predict_endpoint() -> dict:
    request_json = request.get_json()
    ticker = request_json["ticker"]
    print(f"forecasting for: {ticker}")
    last_day, forecast = stocks_forecasting_inference_flow(
        ticker, use_model_registry=False
    )
    # make Date a column instead of the index
    ld = last_day.reset_index()
    fc = forecast.reset_index()

    # turn each row into its own dict of { col: value, … }
    result = {
        "last_day": ld.to_dict(orient="records"),
        "forecast": fc.to_dict(orient="records"),
    }
    return jsonify(result)


# if __name__ == "__main__":
# stocks_forecasting_inference_flow(ticker="AAPL", use_model_registry=False)
# app.run(debug=True, host="0.0.0.0", port=9696)
