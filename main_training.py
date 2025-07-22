import datetime
import os

from data import (
    build_features,
    clean_raw_data,
    create_X_y_multistep,
    load_raw_data,
    sample_tickers_dates,
    split_train_test_panel,
)
from models import create_fit_xgbregressor_chain, evaluate_all

# Configuration parameters
sample_tickers = ["AAPL", "AMZN"]
use_sample_tickers_for_training = True
target = "returns"
forecast_steps = 5

rootpath = os.path.dirname(__file__)
datapath = os.path.join(rootpath, "data")
datasetname = "world-stock-prices-daily-updating"


def run() -> None:
    df_raw = load_raw_data(
        datapath=datapath, user="nelgiriyewithana", datasetname=datasetname
    )
    df_clean = clean_raw_data(df_raw)
    df = sample_tickers_dates(
        df_clean,
        tickers=sample_tickers if use_sample_tickers_for_training else None,
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        clean_sample_fpath_full=None,
        # clean_sample_fpath_full=os.path.join(datapath, f'{datasetname}_clean_sample.csv')
    )
    df_train, df_test = split_train_test_panel(df, train_ratio=0.8)
    df_train_feats, _features2scale = build_features(df_train, lags=3)
    df_test_feats, _features2scale = build_features(df_test, lags=3)
    X_train, y_train = create_X_y_multistep(
        df_train_feats, steps=forecast_steps, target=target
    )
    X_test, y_test = create_X_y_multistep(
        df_test_feats, steps=forecast_steps, target=target
    )

    estimator = create_fit_xgbregressor_chain(X_train, y_train)

    scores = evaluate_all(
        estimator, X_train, y_train, X_test, y_test, df, sample_tickers
    )
    print(scores)


if __name__ == "__main__":
    run()
