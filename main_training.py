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
from scripts.gcp_functions import upload_file_to_folder
from scripts.load_configs import Configs
from utils import get_pipreqs_from_pyproject

# Global parameters
TARGET = "returns"
SAMPLE_TICKERS = ["AAPL", "AMZN"]
ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data")
CONFPATH = os.path.join(ROOTPATH, "config")
ISODATE = datetime.date.today().isoformat()

# Static Parameters that needs to be logged with the model as they will be needed for inference
STATIC_PARS = {
    "lags": 3,  # number of lags to be included
    "steps": 5,  # steps ahead to be forecasted
}

# Set up mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"
CLIENT = MlflowClient()
EXPS, EXP_NAME = build_exp_dicts(os.path.join(CONFPATH, "Exp_CldrFeats_ModReg.yaml"))
mlflow.set_experiment(EXP_NAME)


@flow(name="stocks_forecasting_training_flow")
def stocks_forecasting_training_flow(
    env: str = "prod",
    use_sample_tickers_for_training: bool = True,
    select_only_latest: bool = True,
) -> None:
    """
    The main training workflow for the stocks forecasting project.

    This workflow is orchestrated by prefect and performs the following main and sub-steps:
    1. Base data preparation (task)
        - Get the raw data
        - Clean the raw data
        - Sample tickers and dates
        - Split train and test
    2. Run experiments (task)
        - Prepare features
        - Train the model
        - Evaluate the model
        - Log the model, parameters, metrics to mlfow
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

    assert env in ["test", "dev", "prod"]

    # step1: base data prep (task as sub-flow)
    if env in ["dev", "prod"]:
        if use_sample_tickers_for_training:
            logger.info(
                f"As running in a primary env {env}, use_sample_tickers_for_training is set to False"
            )
            use_sample_tickers_for_training = False
    clean_sample_fdir = os.path.join(DATAPATH, f"cleaned_samples_{env}")
    df, df_train, df_test = base_data_prep(
        env, use_sample_tickers_for_training, clean_sample_fdir
    )

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
    if env in ["dev", "prod"]:
        remove_raw_data(DATAPATH)
        logger.info(f"Running in a primary env {env}, therefore removing raw data")
    else:
        # Not removing the raw data in non-primary envs to enable fast test runs
        logger.info(
            f"Running in a non-primary env {env}, therefore not removing raw data"
        )

    logger.info("Worfklow finalized")


# This is a subflow, calling other tasks
@task(task_run_name="base_data_prep_taskgroup")
def base_data_prep(
    env: str,
    use_sample_tickers_for_training: bool,
    clean_sample_fdir: str | None = None,
) -> tuple:
    # load the raw data
    df_raw, access_date_str = load_raw_data(
        datapath=DATAPATH,
        user="nelgiriyewithana",
        datasetname="world-stock-prices-daily-updating",
    )
    # clean the raw data (e.g. winsorize returns)
    df_clean = clean_raw_data(df_raw)
    # sample tickers and dates
    df, fpath = sample_tickers_dates(
        df_clean,
        tickers=SAMPLE_TICKERS if use_sample_tickers_for_training else None,
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        clean_sample_fdir=clean_sample_fdir,
        access_date_str=access_date_str,
    )
    store_sampled_data_in_gcs(env, fpath)
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
        # Training: prepare data
        CldrFeats = exp["CldrFeats"] if "CldrFeats" in exp.keys() else True
        df_train_feats, _features2scale = build_features(
            df_train, lags=STATIC_PARS["lags"], split="train", CldrFeats=CldrFeats
        )
        df_test_feats, _features2scale = build_features(
            df_test, lags=STATIC_PARS["lags"], split="test", CldrFeats=CldrFeats
        )
        X_train, y_train = create_X_y_multistep(
            df_train_feats, steps=STATIC_PARS["steps"], target=TARGET
        )
        X_test, y_test = create_X_y_multistep(
            df_test_feats, steps=STATIC_PARS["steps"], target=TARGET
        )
        # Instantiate and train a model
        Regularization = exp["ModReg"] if "ModReg" in exp.keys() else True
        estimator = create_fit_xgbregressor_chain(X_train, y_train, Regularization)
        # Evaluate the model
        scores = evaluate_all(
            estimator, X_train, y_train, X_test, y_test, df, SAMPLE_TICKERS
        )
        logger.info(scores)
        # log the parameters
        mlflow.set_tag("run_date", ISODATE)
        mlflow.log_params(exp)
        mlflow.log_params(STATIC_PARS)
        # log the metrics
        for cat, score_dict in scores.items():
            for metric, value in score_dict.items():
                mlflow.log_metrics({f"{cat}_{metric}": value})
        # log the model
        signature = infer_signature(X_train, estimator.predict(X_train))
        pip_reqs = get_pipreqs_from_pyproject(os.path.join(ROOTPATH, "pyproject.toml"))
        mlflow.sklearn.log_model(
            estimator,
            name=MODEL_ARTIFACT_FOLDER,
            pip_requirements=pip_reqs,
            input_example=X_train.head(5),
            signature=signature,
        )


@task(task_run_name="register_best_model")
def store_sampled_data_in_gcs(env: str, fpath: str) -> None:
    logger = get_run_logger()
    config = Configs(env)
    upload_file_to_folder(
        project_id=config.cloud["gcs"]["project"],
        bucket_name=config.cloud["gcs"]["data_monitoring_bucket"],
        folder=f"cleaned_samples_{env}",
        file=fpath,
    )
    logger.info(
        f"Uploaded {fpath} to gs://{config.cloud['gcs']['data_monitoring_bucket']}/cleaned_samples_{env}"
    )


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
    best_run_id = runs[0].info.run_id
    best_model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_FOLDER}"
    logger.info(f"Registering best model with id: {best_run_id}")
    registered = mlflow.register_model(model_uri=best_model_uri, name=REGISTRY_NAME)
    version = registered.version
    logger.info(f"Registered model has version={version}")
    logger.info("Aliasing model as champion")
    CLIENT.set_registered_model_alias(REGISTRY_NAME, alias="champion", version=version)
    logger.info("Tagging model as approved")
    CLIENT.set_model_version_tag(
        REGISTRY_NAME, version=version, key="validation_status", value="approved"
    )


if __name__ == "__main__":
    stocks_forecasting_training_flow(
        env="dev", use_sample_tickers_for_training=True, select_only_latest=True
    )
