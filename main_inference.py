import gzip
import json
import logging
import os
import pickle
import uuid
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, Response, abort, g, jsonify, make_response, request
from mlflow.tracking import MlflowClient
from werkzeug.exceptions import BadRequest, HTTPException
from xgboost import XGBRegressor

from data import (
    build_features,
    clean_raw_data,
    create_X_y_multistep,
)
from raw_data_yf import fetch_ticker_data_from_yf

# Global parameters
SERVICE_VERSION = "forecast-api@0.2.0"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"

rootpath = os.path.dirname(__file__)
MODELPATH = os.path.join(rootpath, "data", "extracted_model")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


def stocks_forecasting_inference_flow(
    ticker: str = "AAPL",
    data_dict: dict | None = None,
    use_model_registry: bool = False,
    past_horizon: int = 1,
) -> None:
    if use_model_registry:
        model, params = retrieve_registered_model()
    else:
        model, params = retrieve_locally_stored_model()
    if data_dict is None:
        df_init = retrieve_ticker_data(ticker)
    else:
        df_init = handle_series_data(data_dict, ticker)
    past, forecast = run_forecast(model, params, df_init, past_horizon=past_horizon)
    logger.info(f"\npast:\n{past}\nforecast:\n{forecast}")
    meta = {  # todo: retrieve missing metadata from registry/storage
        "model_alias": MODEL_ALIAS,
        "model_exp_params": params,
        "model_version": "unknown",
        "model_trained_at": "unknown",
        "model_scores": "unknown",
    }
    return past, forecast, meta


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
    fpath = os.path.join(MODELPATH, "model.pkl")
    logger.info(f"Loading model from: {fpath}")
    with open(fpath, "rb") as f:
        model = pickle.load(f)
    fpath = os.path.join(MODELPATH, "params.json")
    logger.info(f"Loading params from: {fpath}")
    with open(fpath) as f:
        params = json.load(f)
    return model, params


def handle_series_data(data_dict: dict, ticker: str) -> pd.DataFrame:
    dates = data_dict["date"]
    closes = data_dict["close"]
    # Build DataFrame
    df_raw = pd.DataFrame({
        "Date": pd.to_datetime(dates, utc=True, errors="coerce"),
        "Close": closes,
        "Ticker": ticker,
    })
    if df_raw["Date"].isna().any():
        raise ValueError({"Invalid 'date' value(s). Use ISO-8601 like 'YYYY-MM-DD'."})
    # clean the raw data, but do not winsorize
    df_clean = clean_raw_data(df_raw, winsorize=False)
    df_clean.sort_values(["Date"], inplace=True)
    return df_clean


def retrieve_ticker_data(ticker: str) -> pd.DataFrame:
    # download the raw data
    df_raw = fetch_ticker_data_from_yf(ticker=ticker)
    # clean the raw data, but do not winsorize
    df_clean = clean_raw_data(df_raw, winsorize=False)
    df_clean.sort_values(["Date"], inplace=True)
    return df_clean


def run_forecast(
    model: XGBRegressor,
    parameters: dict,
    df_init: pd.DataFrame,
    bizday_offset: bool = True,
    past_horizon: int = 1,
) -> tuple:
    # Build features on recent history
    CldrFeats = parameters["CldrFeats"] if "CldrFeats" in parameters.keys() else "True"
    CldrFeats = True if CldrFeats == "True" else False
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
    returns_series = pd.Series(returns, index=dates, name="Returns")

    # Compute prices from returns
    # Start from last observed close
    ind = 0 - past_horizon
    past_prices = df_init[["Date", "Close"]].iloc[ind:]
    past_prices.set_index("Date", inplace=True)
    past_prices.index.name = None
    price_prev = past_prices.iloc[-1].values[0]
    new_prices = [price_prev]
    for returns in y_hat:
        price_next = price_prev * (1 + returns)
        new_prices.append(price_next)
        price_prev = price_next

    prices_series = pd.Series(new_prices, index=dates, name="Close")

    forecast = pd.concat([returns_series, prices_series], axis=1)

    forecast = forecast.iloc[1:, :]  # exclue the 0th day (included in the past prices)
    return past_prices, forecast


def forecast_from_symbol(ticker: str, past_horizon: int) -> dict:
    past, forecast, meta = stocks_forecasting_inference_flow(
        ticker, use_model_registry=False, past_horizon=past_horizon
    )
    # make Date a column instead of the index
    ld = past.reset_index()
    fc = forecast.reset_index()

    # turn each row into its own dict of { col: value, … }
    result = {
        "past": ld.to_dict(orient="records"),
        "forecast": fc.to_dict(orient="records"),
    }
    return result, meta


def forecast_from_data(ticker: str, past_horizon: int, series: dict) -> dict:
    dates = series.get("date")
    closes = series.get("close")

    # Validate columnar payload
    if not isinstance(series, dict):
        return jsonify({
            "error": "'series' must be an object with 'date' and 'close' arrays"
        }), 400
    if not isinstance(dates, list) or not isinstance(closes, list):
        return jsonify({"error": "'date' and 'close' must be arrays"}), 400
    if len(dates) != len(closes):
        return jsonify({"error": "date/close lengths must match"}), 400
    mindays_tobuild_features = 50
    lookback_mindays = mindays_tobuild_features + past_horizon
    if (
        len(dates) < lookback_mindays
    ):  # models minimum lookback for being able to build all lag and ma features
        return jsonify({
            "error": f"Not enough observations; need at least {lookback_mindays} days of data ({past_horizon} of which past), sent: {len(dates)}."
        }), 422

    data_dict = {"date": dates, "close": closes}

    print("data validated, forecasting..")
    # Run your inference flow that accepts a pre-supplied price series
    past, forecast, meta = stocks_forecasting_inference_flow(
        ticker=ticker,
        data_dict=data_dict,
        use_model_registry=False,
        past_horizon=past_horizon,
    )

    # make Date a column instead of the index
    ld = past.reset_index()
    fc = forecast.reset_index()

    # turn each row into its own dict of { col: value, … }
    result = {
        "past": ld.to_dict(orient="records"),
        "forecast": fc.to_dict(orient="records"),
    }
    return result, meta


def load_json_maybe_compressed(raw: dict, enc: str | None = None) -> dict:
    """
    Supports plain JSON bodies, plus optionally:
      - Content-Encoding: gzip
      - Content-Encoding: br   (if 'brotli' is installed)
    """

    if enc == "gzip":
        try:
            decompressed = gzip.decompress(raw)
        except OSError:
            abort(400, description="Invalid gzip payload")
    elif enc in ("", None):
        # not compressed; leave as-is
        decompressed = raw
    else:
        abort(415, description=f"Unsupported Content-Encoding: {enc}")

    try:
        input_dict = json.loads(decompressed or b"{}")
        return input_dict
    except json.JSONDecodeError:
        abort(400, description="Invalid JSON")


def base_meta() -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "service_version": SERVICE_VERSION,
        "request_id": g.request_id,  # request_id before_request packs request_id in the header, this is just for convenience
    }


def make_api_response(
    data: dict,
    *,  # all args afterwards must be named to avoid mix-ups
    status: int = 200,
    meta: dict | None = None,
    headers: dict | None = None,
    envelope: bool = True,
) -> Response:
    # add base_meta() to meta (right hand side wins on conflict)
    meta_full = base_meta() | (meta or {})

    if envelope:
        body = {"meta": meta_full, "data": data}
    else:
        # legacy shape (v1): keep data at top level
        body = dict(data)  # shallow copy
        body["meta"] = meta_full

    resp = make_response(jsonify(body), status)
    if headers:
        for k, v in headers.items():
            resp.headers[k] = v
    return resp


app = Flask("stocks-forecasting")


@app.before_request
def ensure_request_id() -> None:
    rid = request.headers.get("X-Request-ID")
    if not rid:
        rid = str(uuid.uuid4())
    g.request_id = rid


@app.after_request
def echo_request_id(resp: Response) -> Response:
    # echo on every response
    resp.headers["X-Request-ID"] = g.request_id
    return resp


@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException) -> Response:
    problem = {
        "type": "about:blank",
        "title": e.name,  # e.g. "Bad Request"
        "status": e.code,  # 400
        "detail": e.description,  # e.g. "Missing signature_name"
    }
    resp = jsonify(problem)
    resp.status_code = e.code
    resp.headers["Content-Type"] = "application/problem+json"
    return resp


@app.errorhandler(Exception)
def handle_unexpected(e: Exception) -> Response:
    # last-resort 500 in JSON with stack trace logged
    logger.exception("Unhandled server error")
    resp = jsonify({
        "type": "about:blank",
        "title": "Internal Server Error",
        "status": 500,
        "detail": "An unexpected error occurred.",
    })
    resp.status_code = 500
    resp.headers["Content-Type"] = "application/problem+json"
    return resp


@app.route("/v2/forecast", methods=["POST"])
def forecast_endpoint() -> dict:
    """
    Expects columnar JSON ONLY:
    {
      "signature_name": "from_symbol",  # or "from_data"
      "ticker": "AAPL",
      "series": {  # needed only if "signature_name" = "from_data"
        "date":  ["2025-07-21", "2025-07-22", ...],
        "close": [231.14,       233.02,      ...]
      }
      "past_horizon": 1
    }
    """
    print("forecasting from data for symbol:", end="")
    raw = request.get_data(cache=False)
    enc = (request.headers.get("Content-Encoding") or "").lower()
    p = load_json_maybe_compressed(raw, enc)
    signature_name = p.get("signature_name", "NA")
    ticker = p.get("ticker", "NA")
    past_horizon = p.get("past_horizon", 1)
    print(f"{ticker} with past_horizon: {past_horizon}")
    print(
        f"forecasting for: {ticker} with past_horizon: {past_horizon} via {signature_name} service"
    )

    try:
        if signature_name == "from_symbol":
            result, meta = forecast_from_symbol(ticker, past_horizon)
        elif signature_name == "from_data":
            series = p.get("series") or None
            if series is None:
                print("no series was provided, defaulting to from_symbol service")
                result, meta = forecast_from_symbol(ticker, past_horizon)
            else:
                result, meta = forecast_from_data(ticker, past_horizon, series)
        else:
            abort(400, description=f"Unsupported signature_name: {signature_name}")
    except ValueError as e:
        logger.warning(
            f"Error forecasting for ticker {ticker}",
            extra={"ticker": ticker, "err": str(e)},
        )
        raise BadRequest(description=f"Error forecasting for ticker: {ticker}") from e
        # abort(400, description=f"Error forecasting for ticker: {ticker}")

    meta["api_endpoint"] = "/v2/forecast"
    meta["api_signature_name"] = signature_name
    meta["ticker"] = ticker
    headers = {
        "Cache-Control": "no-store",
        # "Link": '</openapi.json>; rel="describedby"',  # when the documentation is available
    }
    resp = make_api_response(
        result,
        meta=meta,
        headers=headers,
        envelope=True,
    )
    return resp


@app.route("/v1/forecast/from_symbol", methods=["POST"])
def forecast_endpoint_from_symbol() -> dict:
    """
    Expects JSON with the following structure:
    {
      "ticker": "AAPL",
      "past_horizon": 1
    }
    """
    print("forecasting from data for symbol:", end="")
    raw = request.get_data(cache=False)
    enc = (request.headers.get("Content-Encoding") or "").lower()
    p = load_json_maybe_compressed(raw, enc)
    ticker = p.get("ticker", "NA")
    past_horizon = p.get("past_horizon", 1)
    print(f"{ticker} with past_horizon: {past_horizon}")
    result, meta = forecast_from_symbol(ticker, past_horizon)
    meta["api_endpoint"] = "/v1/forecast/from_symbol"
    meta["ticker"] = ticker
    headers = {
        "Cache-Control": "no-store",
        "Deprecation": "true",
        "Link": '</v2/forecast>; rel="successor-version"',
    }
    resp = make_api_response(result, meta=meta, headers=headers, envelope=False)
    return resp


@app.route("/v1/forecast/from_data", methods=["POST"])
def forecast_endpoint_from_series() -> dict:
    """
    Expects columnar JSON ONLY:
    {
      "ticker": "AAPL",
      "series": {
        "date":  ["2025-07-21", "2025-07-22", ...],
        "close": [231.14,       233.02,      ...]
      }
      "past_horizon": 1
    }
    """
    print("forecasting from data for symbol:", end="")
    raw = request.get_data(cache=False)
    enc = (request.headers.get("Content-Encoding") or "").lower()
    p = load_json_maybe_compressed(raw, enc)
    ticker = p.get("ticker", "NA")
    past_horizon = p.get("past_horizon", 1)
    print(f"{ticker} with past_horizon: {past_horizon}")
    series = p.get("series") or None
    if series is None:
        print("no series was provided, defaulting to from_symbol service")
        result, meta = forecast_from_symbol(ticker, past_horizon)
    else:
        result, meta = forecast_from_data(ticker, past_horizon, series)
    meta["api_endpoint"] = "/v1/forecast/from_data"
    meta["ticker"] = ticker
    headers = {
        "Cache-Control": "no-store",
        "X-Request-ID": getattr(g, "request_id", None),
        "Deprecation": "true",
        "Link": '</v2/forecast>; rel="successor-version"',
    }
    resp = make_api_response(result, meta=meta, headers=headers, envelope=False)
    return resp


@app.get("/healthz")
def healthz() -> tuple:
    return jsonify({
        "status": "ok",
        "service": "stocks-forecasting",
        "version": "v1",
        "time": datetime.now(UTC).isoformat(),
    }), 200


@app.get("/")
def index() -> tuple:
    return jsonify({
        "message": "stocks-forecasting API",
        "health": "/healthz",
        "endpoints": [
            {"path": "/v2/forecast", "method": "POST"},
            {"path": "/v1/forecast/from_symbol", "method": "POST"},
            {"path": "/v1/forecast/from_data", "method": "POST"},
        ],
    }), 200


# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=9696)
# forecast_from_symbol(
#     ticker="AAPL", past_horizon=5
# )
