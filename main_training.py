import datetime
import os

import mlflow
import pandas as pd
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

from create_experiments import build_exp_dicts
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
from utils import get_pipreqs_from_pyproject

# Global parameters
TARGET = "returns"
SAMPLE_TICKERS = ["AAPL", "AMZN"]
FORECAST_STEPS = 5
ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data")
CONFPATH = os.path.join(ROOTPATH, "config")
ISODATE = datetime.date.today().isoformat()

# Set up mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
ARTIFACT_PATH = "mlflow_models"
CLIENT = MlflowClient()
EXPS, EXP_NAME = build_exp_dicts(os.path.join(CONFPATH, "Exp_CldrFeats_ModReg.yaml"))
mlflow.set_experiment(EXP_NAME)


@flow(name="stocks_forecasting_training_flow")
def stocks_forecasting_training_flow(
    test_mode: bool = True,
    use_sample_tickers_for_training: bool = True,
    select_only_latest: bool = True,
) -> None:
    """
    The main training workflow for the stocks forecasting project.

    This workflow is orchestrated by prefect and performs the following main and sub-steps:
    1. Base data preparation (task group)
      - Get the raw data (task)
      - Clean the raw data  (task)
      - Sample tickers and dates (task)
      - Split train and test
    2. Run experiments (task group)
      - Prepare train features
      - Start training the model asynchronously
      - Prepare test data while the model is training
      - Once the estimator and data are ready, evaluate the model
    3. Register the best model (task)
    4. Cleanup (task)

    Args:
    test_mode: bool
        whether to run the pipeline in test mode or not
    use_sample_tickers_for_training: bool
        whether to use a sample of tickers for training
    select_only_latest: bool
        whether to select only the latest model under the experiments, i.e., from today
    """
    logger = get_run_logger()

    # step1: base data prep (task as sub-flow)
    df, df_train, df_test = base_data_prep(use_sample_tickers_for_training)

    # step 2: run experiments (tasks as sub-flow)
    for i, exp in enumerate(EXPS):
        logger.info(f"Running experiment {i + 1} of {len(EXPS)}")
        # create a run name based on experiment parameters:
        run_name = "_".join([f"{key}={value}" for key, value in exp.items()])
        run_single_experiment(
            exp=exp, run_name=run_name, df=df, df_train=df_train, df_test=df_test
        )

    # step 3: register_best_model (task)
    register_best_model(only_latest=select_only_latest)

    # step 4: cleanup (task)
    if not test_mode:
        remove_raw_data(DATAPATH)
    else:
        logger.info("Running in test mode, therefore not removing raw data")

    logger.info("Worfklow finalized")


# This is a subflow, calling other tasks
@task(task_run_name="base_data_prep_taskgroup")
def base_data_prep(use_sample_tickers_for_training: bool) -> tuple:
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
        tickers=SAMPLE_TICKERS if use_sample_tickers_for_training else None,
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        clean_sample_fpath_full=None,
    )
    # Split train and test
    df_train, df_test = split_train_test_panel(df, train_ratio=0.8)
    return df, df_train, df_test


# This is a subflow, calling other tasks
@task(task_run_name="experiment_{run_name}")
def run_single_experiment(
    exp: dict,
    run_name: str,
    df: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> str:
    logger = get_run_logger()
    run_name_root = "_".join([f"{key}={value}" for key, value in exp.items()])
    run_name = f"{run_name_root}"
    print(f"Runnning experiment: {run_name} with config: {exp}")
    # if somehow a stray run is active, close it
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("run_date", ISODATE)
        mlflow.log_params(exp)

        # Training: prepare train data and start training the model asynchronously
        CldrFeats = exp["CldrFeats"] if "CldrFeats" in exp.keys() else True
        build_features_train_task = build_features.submit(
            df_train, lags=3, split="train", CldrFeats=CldrFeats
        )
        df_train_feats, _features2scale = build_features_train_task.result()
        create_X_y_multistep_train_task = create_X_y_multistep.submit(
            df_train_feats, steps=FORECAST_STEPS, target=TARGET, split="train"
        )
        # Instantiate and train a model
        X_train, y_train = create_X_y_multistep_train_task.result()
        Regularization = exp["ModReg"] if "ModReg" in exp.keys() else True
        create_fit_xgbregressor_chain_task = create_fit_xgbregressor_chain.submit(
            X_train, y_train, Regularization
        )

        # Evaluate the model: prepare test data while the model is training
        build_features_test_task = build_features.submit(
            df_test, lags=3, split="test", CldrFeats=CldrFeats
        )
        df_test_feats, _features2scale = build_features_test_task.result()
        create_X_y_multistep_test_task = create_X_y_multistep.submit(
            df_test_feats, steps=FORECAST_STEPS, target=TARGET, split="test"
        )
        X_test, y_test = create_X_y_multistep_test_task.result()
        estimator = create_fit_xgbregressor_chain_task.result()
        # Once the estimator and data are ready, evaluate the model
        scores = evaluate_all(
            estimator, X_train, y_train, X_test, y_test, df, SAMPLE_TICKERS
        )
        logger.info(scores)

        for cat, score_dict in scores.items():
            for metric, value in score_dict.items():
                mlflow.log_metrics({f"{cat}_{metric}": value})

        signature = infer_signature(X_train, estimator.predict(X_train))
        pip_reqs = get_pipreqs_from_pyproject(os.path.join(ROOTPATH, "pyproject.toml"))
        mlflow.sklearn.log_model(
            estimator,
            name=ARTIFACT_PATH,
            pip_requirements=pip_reqs,
            input_example=X_train.head(5),
            signature=signature,
        )

        return


@task(task_run_name="register_best_model")
def register_best_model(only_latest: bool = True) -> None:
    """Select best run by RMSE and register its model."""
    logger = get_run_logger()
    # pull all runs under the experiment
    experiment = CLIENT.get_experiment_by_name(EXP_NAME)
    logger.info(
        f"Searching best model for experiment: {EXP_NAME} with id: {experiment.experiment_id}"
    )
    if only_latest:
        logger.info("Selecting only the latest model from todays runs")
        filter_string = f"tags.run_date = '{ISODATE}'"
    else:
        logger.info("Selecting the best model from all runs")
        filter_string = None
    runs = CLIENT.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        filter_string=filter_string,
        max_results=2,
        order_by=["metrics.overall_test_rmse ASC"],
    )
    model_name = "stocks_forecasting_regressor_candidates"
    best_run_id = runs[0].info.run_id
    best_model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
    logger.info(f"Registering best model with id: {best_run_id}")
    registered = mlflow.register_model(model_uri=best_model_uri, name=model_name)
    version = registered.version
    logger.info(f"Registered model has version={version}")
    logger.info("Aliasing model as champion")
    CLIENT.set_registered_model_alias(model_name, alias="champion", version=version)
    logger.info("Tagging model as approved")
    CLIENT.set_model_version_tag(
        model_name, version=version, key="validation_status", value="approved"
    )


if __name__ == "__main__":
    stocks_forecasting_training_flow(
        test_mode=True, use_sample_tickers_for_training=True
    )
