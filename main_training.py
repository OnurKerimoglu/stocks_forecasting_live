import datetime
import os

from prefect import Flow, get_run_logger

from data import (
    build_features,
    clean_raw_data,
    create_X_y_multistep,
    load_raw_data,
    remove_raw_data,
    sample_tickers_dates,
    split_train_test_panel,
)
from models import create_fit_xgbregressor_chain, evaluate_all


@Flow
def stocks_forecasting_training_flow(
    test_mode: bool = True,
    use_sample_tickers_for_training: bool = True
    ) -> None:
    """
    The main training workflow for the stocks forecasting project.

    This workflow is orchestrated by prefect and performs the following steps:
    1. Get the raw data
    2. Clean the raw data (e.g. winsorize returns)
    3. Sample tickers and dates
    4. Split train and test
    5. Training: prepare train data
    6. Start training the model asynchronously
    7. Prepare test data while the model is training
    8. Once the estimator and data are ready, evaluate the model
    9. Cleanup (remove raw data if not in test mode)

    Args:
    test_mode: bool
        whether to run the pipeline in test mode or not
    use_sample_tickers_for_training: bool
        whether to use a sample of tickers for training
    """
    logger = get_run_logger()
    # Configuration parameters
    target = "returns"
    sample_tickers = ["AAPL", "AMZN"]
    forecast_steps = 5
    rootpath = os.path.dirname(__file__)
    datapath = os.path.join(rootpath, "data")

    # get the raw data
    df_raw = load_raw_data(
        datapath=datapath, user="nelgiriyewithana", datasetname="world-stock-prices-daily-updating"
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

    # cleanup
    if not test_mode:
        remove_raw_data(datapath)
    else:
        logger.info("Running in test mode, therefore not removing raw data")


if __name__ == "__main__":
    stocks_forecasting_training_flow(
        test_mode=True,
        use_sample_tickers_for_training=True
    )
