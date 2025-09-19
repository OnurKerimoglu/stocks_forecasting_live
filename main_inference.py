import gzip
import json
import logging
import os
import uuid
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, Response, abort, g, jsonify, make_response, request
from pandas.tseries.offsets import BDay
from werkzeug.exceptions import BadRequest, HTTPException
from xgboost import XGBRegressor

from gcp_functions import blob_exists, load_json_from_gcs
from load_configs import Configs
from raw_data_yf import fetch_ticker_data_from_yf
from utils import resolve_model_bundle_uri_for_env

# Global parameters
SERVICE_VERSION = "forecast-api@0.2.0"
ROOTPATH = os.path.dirname(__file__)
MODELPATH = os.path.join(ROOTPATH, "models")
MLFLOWPATH = os.path.join(MODELPATH, "mlflow_runs")
EXTRACTED_MODEL_DIRNAME = "extracted_model"
MODEL_ARTIFACT_FOLDER = "mlflow_models"
# Derived global parameters
config_cloud = Configs().cloud
GCP_PROJECT = config_cloud["gcs"]["project"]
GCP_BUCKET = config_cloud["gcs"]["mlflow_bucket"]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


def retrieve_model_from_gcs(model_id: str, env: str = "prod") -> tuple:
    if model_id == "default":
        # read the model uri from promotion state
        bundle_uri, manifest = resolve_model_bundle_uri_for_env(
            env=env,
            gcp_project=GCP_PROJECT,
            gcp_bucket=GCP_BUCKET,
            status_blob="promotion_status.json",
            local_status_path=os.path.join(MODELPATH, "promotion_status.json"),
        )
        logger.info(
            f"from gcs load default {env} model pointed by promotion_status: {manifest}"
        )
    else:
        bundle_uri = f"gs://{GCP_BUCKET}/runs/{model_id}"
        logger.info(f"from gcs load requested model: {model_id}")

    if not blob_exists(GCP_PROJECT, GCP_BUCKET, bundle_uri):
        raise FileNotFoundError(f"{bundle_uri} was not found")

    model_uri = f"{bundle_uri}/{MODEL_ARTIFACT_FOLDER}"
    loaded_pipeline = mlflow.sklearn.load_model(model_uri)
    model_dirname = model_uri.split(GCP_BUCKET + "/")[1]
    metadata = load_json_from_gcs(
        GCP_PROJECT, GCP_BUCKET, f"{model_dirname}/metadata.json"
    )
    return loaded_pipeline, metadata


def retrieve_local_model(model_id: str) -> tuple:
    if model_id == "default":
        print(f"loading default model from the local {EXTRACTED_MODEL_DIRNAME}")
        mlmodeldir = os.path.join(
            MODELPATH, EXTRACTED_MODEL_DIRNAME, MODEL_ARTIFACT_FOLDER
        )
    else:
        mlmodeldir = os.path.join(MLFLOWPATH, model_id, MODEL_ARTIFACT_FOLDER)
    if not os.path.exists(mlmodeldir):
        raise FileNotFoundError(f"{mlmodeldir} was not found")
    loaded_pipeline = mlflow.sklearn.load_model(mlmodeldir)
    metadata_fpath = os.path.join(mlmodeldir, "metadata.json")
    with open(metadata_fpath) as f:
        metadata = json.load(f)
    return loaded_pipeline, metadata


def retrieve_model(model_id: str, env: str | None = "prod") -> tuple:
    try:
        return retrieve_local_model(model_id)
    except FileNotFoundError:
        return retrieve_model_from_gcs(model_id, env=env)


# cache default model, params and metadata
def_model, def_metadata = retrieve_model(
    model_id="default",  # will try to load from local EXTRACTED_MODEL_DIRNAME
    env="prod",  # if the local load fails, load from GCS, (if model_id=default, model pointed by promotion_status)
)
logger.info(f"cached default model info: {def_metadata}")


def stocks_forecasting_inference_flow(
    ticker: str = "AAPL",
    data_dict: dict | None = None,
    past_horizon: int = 1,
    model_id: str = "default",
) -> None:
    model, metadata = fetch_model(model_id)
    meta_summary = summarize_metadata(metadata)
    if data_dict is None:
        df_init = retrieve_ticker_data(ticker)
    else:
        df_init = handle_series_data(data_dict, ticker)
    past, forecast = run_forecast(model, df_init, past_horizon=past_horizon)
    logger.info(f"\nmodel info:{meta_summary}\npast:\n{past}\nforecast:\n{forecast}")
    return past, forecast, meta_summary


def fetch_model(model_id: str) -> tuple:
    if model_id == "default":
        # use the cached model
        logger.info("Using cached model")
        model, metadata = def_model, def_metadata
    else:
        # retrieve the model
        logger.info(f"Requested to use alternative model: {model_id}")
        model, metadata = retrieve_model(model_id)
    return model, metadata


def summarize_metadata(metadata: dict) -> dict:
    metadata_summary = {
        "model_registry_name": metadata["registry_name"],
        "model_alias": metadata["model_alias"],
        "model_version": metadata["version"],
        "model_run_id": metadata["run_id"],
        "model_uri": metadata["run_info"]["_artifact_uri"],
        "model_commit_id": metadata["tags"]["mlflow.source.git.commit"],
        "model_trained_at": metadata["tags"]["run_date"],
        "params": metadata["params"],
    }
    return metadata_summary


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
    df_raw.sort_values(["Date"], inplace=True)
    return df_raw


def retrieve_ticker_data(ticker: str) -> pd.DataFrame:
    # download the raw data
    df_raw = fetch_ticker_data_from_yf(ticker=ticker)
    df_raw.sort_values(["Date"], inplace=True)
    return df_raw


def run_forecast(
    pp_model_pl: XGBRegressor,
    df_init: pd.DataFrame,
    bizday_offset: bool = True,
    past_horizon: int = 1,
) -> tuple:
    # Build features on recent history
    X_init, y_init = pp_model_pl.make_X_y(df_init)

    # Grab the last row of features (drop identifiers)
    X_step = X_init.iloc[[-1], :]

    # Predict returns
    y_hat = pp_model_pl.estimator_.predict(X_step)[0]

    # Generate the timestamps (business days)
    last_date = df_init["Date"].max()
    steps = len(y_hat)
    if bizday_offset:
        dates_ts = [last_date + BDay(i) for i in range(0, steps + 1)]
    else:
        dates_ts = [last_date + pd.Timedelta(days=i) for i in range(0, steps + 1)]
    dates = [timestamp.date() for timestamp in dates_ts]
    last_return = y_init.iloc[-1].values[0]
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


def forecast_from_symbol(
    ticker: str, past_horizon: int, model_id: str = "default"
) -> dict:
    past, forecast, meta = stocks_forecasting_inference_flow(
        ticker, past_horizon=past_horizon, model_id=model_id
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


def forecast_from_data(
    ticker: str, past_horizon: int, series: dict, model_id: str = "default"
) -> dict:
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
        ticker=ticker, data_dict=data_dict, past_horizon=past_horizon, model_id=model_id
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
      "past_horizon": 1,
      "model_id": "abc123",
    }
    """
    print("forecasting from data for symbol:", end="")
    raw = request.get_data(cache=False)
    enc = (request.headers.get("Content-Encoding") or "").lower()
    p = load_json_maybe_compressed(raw, enc)
    signature_name = p.get("signature_name", "NA")
    ticker = p.get("ticker", "NA")
    past_horizon = p.get("past_horizon", 1)
    model_id = p.get("model_id", "default")
    print(f"{ticker} with past_horizon: {past_horizon}; will use model_id: {model_id}")
    print(
        f"forecasting for: {ticker} with past_horizon: {past_horizon} via {signature_name} service"
    )

    try:
        if signature_name == "from_symbol":
            result, meta = forecast_from_symbol(ticker, past_horizon, model_id)
        elif signature_name == "from_data":
            series = p.get("series") or None
            if series is None:
                print("no series was provided, defaulting to from_symbol service")
                result, meta = forecast_from_symbol(ticker, past_horizon, model_id)
            else:
                result, meta = forecast_from_data(
                    ticker, past_horizon, series, model_id
                )
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
    model_id = p.get("model_id", "default")
    print(f"{ticker} with past_horizon: {past_horizon}; will use model_id: {model_id}")
    result, meta = forecast_from_symbol(ticker, past_horizon, model_id)
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
    model_id = p.get("model_id", "default")
    print(f"{ticker} with past_horizon: {past_horizon}; will use model_id: {model_id}")
    series = p.get("series") or None
    if series is None:
        print("no series was provided, defaulting to from_symbol service")
        result, meta = forecast_from_symbol(ticker, past_horizon, model_id)
    else:
        result, meta = forecast_from_data(ticker, past_horizon, series, model_id)
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
# stocks_forecasting_inference_flow(
#     ticker="AAPL",
#     data_dict=None,
#     past_horizon=1,
#     # model_id="default"
#     model_id="fb7f10cfb2b547c4ab33a1e7ee4e3d91"
# )
