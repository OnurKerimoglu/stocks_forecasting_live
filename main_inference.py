import datetime
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

# from data import build_features, create_X_y_multistep
from data import (
    build_features,
    clean_raw_data,
    create_X_y_multistep,
    load_raw_data,
    sample_tickers_dates,
    split_train_test_panel,
)

# Global parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"


def main(ticker: str) -> None:
    model, params = retrieve_registered_model()
    df_init = retrieve_init_data_temp(ticker)
    last_day, forecast = direct_multistep_forecast(model, params, df_init)
    print(f"last day:\n{last_day}\nforecast:\n{forecast}")


def retrieve_registered_model() -> tuple:
    # Load the model from the Model Registry
    model_uri = f"models:/{REGISTRY_NAME}@{MODEL_ALIAS}"
    print("Retrieveing model_uri: ", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    mv = CLIENT.get_model_version_by_alias(name=REGISTRY_NAME, alias=MODEL_ALIAS)
    run = CLIENT.get_run(mv.run_id)
    params = run.data.params

    return model, params


def retrieve_init_data_temp(ticker: str) -> pd.DataFrame:
    ROOTPATH = os.path.dirname(__file__)
    DATAPATH = os.path.join(ROOTPATH, "data")
    # load the raw data
    df_raw = load_raw_data(
        datapath=DATAPATH,
        user="nelgiriyewithana",
        datasetname="world-stock-prices-daily-updating",
    )
    # clean the raw data (e.g. winsorize returns)
    df_clean = clean_raw_data(df_raw)
    # sample tickers and dates
    df = sample_tickers_dates(
        df_clean,
        tickers=[ticker],
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        clean_sample_fpath_full=None,
    )
    # Split train and test
    _df_train, df_test = split_train_test_panel(df, train_ratio=0.8)
    df_test_raw_last = df_test[df_test["Ticker"] == ticker].copy()
    return df_test_raw_last


def direct_multistep_forecast(
    model: XGBRegressor,
    parameters: dict,
    df_init: pd.DataFrame,
    bizday_offset: bool = True,
) -> tuple:
    # Build features on your last observed history
    df_feats, _ = build_features(
        df_init, lags=int(parameters["lags"]), CldrFeats=parameters["CldrFeats"]
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

    return result.iloc[[0], :], result.iloc[1:, :]


if __name__ == "__main__":
    main(ticker="AAPL")
