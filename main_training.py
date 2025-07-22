import datetime
import os

from prefect import Flow, get_run_logger

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


@Flow
def stocks_forecasting_training_pipeline() -> None:
    logger = get_run_logger()
    # get the raw data
    df_raw = load_raw_data(
        datapath=datapath, user="nelgiriyewithana", datasetname=datasetname
    )
    # clean the raw data (e.g. winsorize returns)
    df_clean = clean_raw_data(df_raw)
    # sample tickers and dates
    df = sample_tickers_dates(
        df_clean,
        tickers=sample_tickers if use_sample_tickers_for_training else None,
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        clean_sample_fpath_full=None
    )
    # Split train and test
    df_train, df_test = split_train_test_panel(df, train_ratio=0.8)
    # Training: prepare train data and start training the model asynchronously
    build_features_train_task = build_features.submit(df_train, lags=3, split="train")
    df_train_feats, _features2scale = build_features_train_task.result()
    create_X_y_multistep_train_task = create_X_y_multistep.submit(
        df_train_feats, steps=forecast_steps, target=target, split="train"
    )
    # Instantiate and train a model
    X_train, y_train = create_X_y_multistep_train_task.result()
    create_fit_xgbregressor_chain_task = create_fit_xgbregressor_chain.submit(X_train, y_train)

    # Evaluate the model: prepare test data while the model is training
    build_features_test_task = build_features.submit(df_test, lags=3, split="test")
    df_test_feats, _features2scale = build_features_test_task.result()
    create_X_y_multistep_test_task = create_X_y_multistep.submit(
        df_test_feats, steps=forecast_steps, target=target, split="test"
    )
    X_test, y_test = create_X_y_multistep_test_task.result()
    estimator = create_fit_xgbregressor_chain_task.result()
    # Once the estimator and data are ready, evaluate the model
    scores = evaluate_all(
        estimator, X_train, y_train, X_test, y_test, df, sample_tickers
    )
    logger.info(scores)


if __name__ == "__main__":
    stocks_forecasting_training_pipeline()
    # stocks_forecasting_training_pipeline.deploy(
    #     name="stocks_forecasting_train",
    #     work_pool_name="stocks_forecasting_live_local",
    #     image="my-image",
    #     push=False,
    #     # cron="* * * * *",
    # )
