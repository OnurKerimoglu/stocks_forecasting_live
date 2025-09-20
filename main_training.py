import datetime
import json
import os

import mlflow
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

from create_experiments import build_exp_dicts
from data import split_train_test_panel
from gcp_functions import clear_gcs_folder, upload_directory, upload_file_to_folder
from load_configs import Configs
from mlflow_helpers import MLFlowRetriever
from models import evaluate_all
from preprocessor_model_pipeline import PpModelPl
from raw_data import (
    load_raw_data,
    remove_raw_data,
    sample_tickers_dates,
)
from utils import get_pipreqs_from_pyproject

# Global parameters
TARGET = "returns"
SAMPLE_TICKERS = ["AAPL", "AMZN"]
ROOTPATH = os.path.dirname(__file__)
DATAPATH = os.path.join(ROOTPATH, "data")
RAWDATAPATH = os.path.join(DATAPATH, "raw")
SAMPLEPATHROOT = "raw_samples"
MODELPATH = os.path.join(ROOTPATH, "models")
MLFLOWPATH = os.path.join(MODELPATH, "mlflow_runs")
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
REGISTRY_NAME = "stocks_forecasting"
MODEL_ALIAS = "Candidate"
CLIENT = MlflowClient()
EXPS, EXP_NAME = build_exp_dicts(
    os.path.join(CONFPATH, "Exp_CldrFeats_ModReg.yaml")  # _test
)
mlflow.set_experiment(EXP_NAME)


@flow(name="stocks_forecasting_training_flow")
def stocks_forecasting_training_flow(
    env: str = "prod",
    datasource: str = "yahoofinance",
    use_sample_tickers_for_training: bool = True,
    select_only_latest: bool = True,
) -> None:
    """
    The main training workflow for the stocks forecasting project.

    This workflow is orchestrated by prefect and performs the following main and sub-steps:
    1. Base data preparation (task)
        - Get the raw data
        - Sample tickers and dates
        - Store sample data in gcs
        - Split train and test
    2. Run experiments (task)
        - Initialize a preporocessor-model pipeline based on exp pars
        - Fit the model (i.e., prepare features and train the model)
        - Evaluate the model
        - Log the model, parameters, metrics to mlfow
    3. Register the best model (task)
    4. Export the best model to local MLFLOWPATH (task)
    5. Uload the best model to GCP (task)
    6. Cleanup (task)

    Args:
    test_mode: bool
        whether to run the pipeline in test mode or not
    use_sample_tickers_for_training: bool
        whether to use a sample of tickers for training
    select_only_latest: bool
        whether to select only the latest model under the experiments, i.e., from today
    """
    logger = get_run_logger()
    try:
        assert env in ["test", "dev", "prod"]
    except Exception as err:
        raise ValueError(
            f"env must be one of ['test', 'dev', 'prod'], but got {env}"
        ) from err

    # step1: base data prep (task as sub-flow)
    if env in ["dev", "prod"]:
        if use_sample_tickers_for_training:
            logger.info(
                f"As running in a primary env {env}, use_sample_tickers_for_training is set to False"
            )
            use_sample_tickers_for_training = False
    sample_fdir = os.path.join(DATAPATH, f"{SAMPLEPATHROOT}_{env}")
    df, df_train, df_test = base_data_prep(
        datasource, env, use_sample_tickers_for_training, sample_fdir
    )

    # step 2: run experiments (task)
    for i, exp in enumerate(EXPS):
        logger.info(f"Running experiment {i + 1} of {len(EXPS)}")
        # create a run name based on experiment parameters:
        run_name = "_".join([f"{key}={value}" for key, value in exp.items()])
        run_single_experiment(
            exp=exp, run_name=run_name, df=df, df_train=df_train, df_test=df_test
        )

    # step 3: register_best_model (task)
    best_run_id = register_best_model(only_latest=select_only_latest)

    # step 4: export best model (task)
    export_model(best_run_id)

    # step 5: upload the best model directory cloud
    upload_model_to_gcs(env, best_run_id)

    # step 6: cleanup (task)
    if env in ["dev", "prod"]:
        remove_raw_data(RAWDATAPATH, datasource)
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
    datasource: str,
    env: str,
    use_sample_tickers_for_training: bool,
    sample_fdir: str | None = None,
) -> tuple:
    # load the raw data
    df_raw, access_date_str = load_raw_data(
        datasource=datasource,
        datapath=DATAPATH,
        localrun=True,
        env=env,
        user="nelgiriyewithana",
        datasetname="world-stock-prices-daily-updating",
    )
    # sample tickers and dates
    df, fpath = sample_tickers_dates(
        df_raw,
        tickers=SAMPLE_TICKERS if use_sample_tickers_for_training else None,
        startdate=datetime.datetime.now() - datetime.timedelta(days=365 * 5 + 1),
        datasource=datasource,
        sample_fdir=sample_fdir,
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
    logger.info(f"Runnning experiment: {run_name} with config: {exp}")
    # if somehow a stray run is active, close it
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=run_name) as run:
        # Parse parameters
        CldrFeats = exp["CldrFeats"] if "CldrFeats" in exp.keys() else True
        ModReg = exp["ModReg"] if "ModReg" in exp.keys() else True
        pp_model_pl = PpModelPl(
            target="returns",
            steps=STATIC_PARS["steps"],
            date_col="Date",
            price_col="Close",
            ticker_col="Ticker",
            winsorize=True,
            q_low=0.01,
            q_high=0.99,
            lags=STATIC_PARS["lags"],
            CldrFeats=CldrFeats,
            ModReg=ModReg,
        )
        logger.info("Training the model..")
        pp_model_pl.fit(df_train)
        logger.info("Training finalized.")
        # Evaluate the model
        X_train, y_train = pp_model_pl.make_X_y(df_train)
        X_test, y_test = pp_model_pl.make_X_y(df_test)
        scores = evaluate_all(
            pp_model_pl.estimator_, X_train, y_train, X_test, y_test, df, SAMPLE_TICKERS
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
        signature = infer_signature(df_train, pp_model_pl.predict(df_train))
        pip_reqs = get_pipreqs_from_pyproject(os.path.join(ROOTPATH, "pyproject.toml"))
        # serialized the pipeline as a sklearn-flavor model and log it
        info = mlflow.sklearn.log_model(
            sk_model=pp_model_pl,
            name=MODEL_ARTIFACT_FOLDER,
            pip_requirements=pip_reqs,
            input_example=df_train.head(80),
            signature=signature,
            code_paths=[
                "./src/preprocessor_model_pipeline.py",  # file defining PpModelPl
                "./src/data.py",  # module providing build_features, create_X_y_multistep
                "./src/models.py",  # module providing create_xgbregressor_chain
            ],
        )
        logger.info(f"Logged run ID: {run.info.run_id}")
        logger.info(f"Logged model URI: {info.model_uri}")


@task(task_run_name="register_best_model")
def store_sampled_data_in_gcs(env: str, fpath: str) -> None:
    logger = get_run_logger()
    config = Configs(env)
    upload_file_to_folder(
        project_id=config.cloud["gcs"]["project"],
        bucket_name=config.cloud["gcs"]["data_monitoring_bucket"],
        folder=f"{SAMPLEPATHROOT}_{env}",
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
    CLIENT.set_registered_model_alias(REGISTRY_NAME, alias=MODEL_ALIAS, version=version)
    logger.info("Tagging model as approved")
    CLIENT.set_model_version_tag(
        REGISTRY_NAME, version=version, key="validation_status", value="approved"
    )
    return best_run_id


@task(task_run_name="export_best_model")
def export_model(run_id: str) -> None:
    """Export run to local filesystem"""
    logger = get_run_logger()
    logger.info(f"Exporting model with run_id: {run_id}")
    localdir = os.path.join(MLFLOWPATH, run_id)
    if os.path.exists(localdir):
        logger.info(f"Model had been ealier exported to: {localdir}")
        return
    # if the directory does not exist, download the artifacts
    logger.info(f"Exporting model to: {localdir}")
    os.makedirs(localdir)
    download_artifacts(
        artifact_uri=f"runs:/{run_id}/{MODEL_ARTIFACT_FOLDER}", dst_path=localdir
    )
    # collect and store additional metadata
    run_id_meta, metadata = MLFlowRetriever(
        client=CLIENT
    ).retrieve_mlflow_model_metadata(
        registry_name=REGISTRY_NAME, model_alias=MODEL_ALIAS
    )
    if run_id != run_id_meta:
        msg = f"run_id: {run_id} does not match the run_id of the retrieved metadata: {run_id_meta}"
        logger.error(msg)
        raise RuntimeError(msg)
    with open(os.path.join(localdir, MODEL_ARTIFACT_FOLDER, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"model artifacts are stored in: {localdir}")


@task(task_run_name="upload_to_gcs")
def upload_model_to_gcs(env: str, run_id: str) -> None:
    """
    Syncing a local directory to a GCS Bucket.

    Parameters:
        localpath (str): Path to the local directory.
        configs (dict): Configuration dict with GCS project and bucket.
    """
    logger = get_run_logger()
    logger.info(f"Exporting best model with run_id: {run_id}")
    localdir = os.path.join(MLFLOWPATH, run_id)
    if not os.path.isdir(localdir):
        raise FileNotFoundError(f"Local directory '{localdir}' does not exist.")
    configs = Configs(env)
    project_id = configs.cloud["gcs"]["project"]
    bucket_name = configs.cloud["gcs"]["mlflow_bucket"]
    gcs_folder = "runs/" + os.path.basename(localdir.rstrip("/"))

    clear_gcs_folder(project_id, bucket_name, gcs_folder)
    upload_directory(project_id, bucket_name, gcs_folder, localdir)


if __name__ == "__main__":
    stocks_forecasting_training_flow(
        env="test",
        datasource="yahoofinance",
        use_sample_tickers_for_training=True,
        select_only_latest=True,
    )
