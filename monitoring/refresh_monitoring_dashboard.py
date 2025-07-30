import argparse
import datetime
import logging
import os
import random
import time
import warnings
from json import load as jload
from pickle import load as pload

import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift

from data import (
    build_features,
    create_X_y_multistep,
)
from scripts.gcp_functions import (
    load_json_from_gcs,
    load_pickle_from_gcs,
    read_file_as_df,
)
from scripts.load_configs import Configs

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

ROOTPATH = os.path.dirname(os.path.dirname(__file__))
DATAPATH = os.path.join(ROOTPATH, "data")
SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists evidently_metrics;
create table evidently_metrics(
	timestamp timestamp,
    num_tickers integer,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=admin"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=stocks"


def load_ref_data_model_params(
    configs: dict | None = None, prefix: str = "ref_data_model"
) -> tuple:
    if configs is not None:
        bucket = configs.cloud["gcs"]["data_monitoring_bucket"]
        logging.info(f"Loading ref data, model and params from GCS bucket: {bucket}")
        # Read ref_data, ref_params and ref_estimator from GCS
        ref_data = read_file_as_df(
            configs.cloud["gcs"]["project"],
            configs.cloud["gcs"]["data_monitoring_bucket"],
            f"{prefix}/data.parquet",
        )
        ref_params = load_json_from_gcs(
            configs.cloud["gcs"]["project"],
            configs.cloud["gcs"]["data_monitoring_bucket"],
            f"{prefix}/params.json",
        )
        ref_estimator = load_pickle_from_gcs(
            configs.cloud["gcs"]["project"],
            configs.cloud["gcs"]["data_monitoring_bucket"],
            f"{prefix}/model.pkl",
        )
    else:
        # Read from the local filesystem
        ref_path = os.path.join(ROOTPATH, prefix)
        if not os.path.exists(ref_path):
            raise Exception(
                f"no config provided for GCS and local path {ref_path} does not exist"
            )
        else:
            fpath = os.path.join(ref_path, "data.parquet")
            ref_data = pd.read_parquet(fpath)
            logging.info(f"Loading new data from filesystem: {fpath}")
            with open(os.path.join(ref_path, "params.json")) as f:
                ref_params = jload(f)
            with open(os.path.join(ref_path, "model.pkl"), "rb") as f:
                ref_estimator = pload(f)
    # Convert the multi-index df to single-index
    ref_data_flat = ref_data.reset_index(level="Ticker")
    return ref_data_flat, ref_params, ref_estimator


def load_new_data(configs: dict, env: str, fname: str) -> pd.DataFrame:
    if configs is not None:
        # Load the df from GCS
        data_path = {
            "project": configs.cloud["gcs"]["project"],
            "bucket": configs.cloud["gcs"]["data_monitoring_bucket"],
            "prefix": f"cleaned_samples_{env}",
            "fname": fname,
        }
        bucket = data_path["bucket"]
        logging.info(f"Loading new data from GCS bucket: {bucket}")
        df = read_file_as_df(
            data_path["project"],
            data_path["bucket"],
            data_path["prefix"] + "/" + data_path["fname"],
        )
    else:
        # Read from the local filesystem
        fpath = os.path.join(DATAPATH, f"sample_cleaned_samples_{env}", fname)
        logging.info(f"Loading new data from filesystem: {fpath}")
        if not os.path.exists(fpath):
            raise Exception(
                f"no config provided for GCS and local path {fpath} does not exist"
            )
        else:
            df = pd.read_parquet(fpath)
    return df


def prepare_new_data(
    configs: dict, env: str, fname: str, params: dict, estimator: object
) -> pd.DataFrame:
    # load the df
    df = load_new_data(configs, env, fname)
    TARGET = "returns"
    df_feats, _features2scale = build_features(
        df, lags=int(params["lags"]), split="train", CldrFeats=params["CldrFeats"]
    )
    X, y = create_X_y_multistep(df_feats, steps=int(params["steps"]), target=TARGET)
    y_hat = estimator.predict(X)
    data = X.copy()
    data["target"] = y["y_step_1"]
    data["prediction"] = y_hat[:, 0]
    # Convert multi-index df to single index
    data_flat = data.reset_index(level="Ticker")
    return data_flat


def prep_db() -> None:
    with psycopg.connect(CONNECTION_STRING, autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='stocks'")
        if len(res.fetchall()) == 0:
            conn.execute("create database stocks;")
        with psycopg.connect(
            "host=localhost port=5432 dbname=stocks user=postgres password=admin"
        ) as conn:
            conn.execute(create_table_statement)


def calculate_metrics_postgresql(
    i: int,
    new_data: pd.DataFrame,
    ref_data: pd.DataFrame,
    data_definition: DataDefinition,
    report: Report,
) -> None:
    days_ago = i
    current_date = new_data.index.unique()[-1 - days_ago]
    logging.info(f"Processing: {current_date}")
    current_data = new_data[new_data.index == current_date]
    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(ref_data, data_definition=data_definition)

    run = report.run(reference_data=reference_dataset, current_data=current_dataset)
    result = run.dict()

    num_tickers = len(current_data.Ticker.values)
    prediction_drift = result["metrics"][0]["value"]
    num_drifted_columns = result["metrics"][1]["value"]["count"]
    share_missing_values = result["metrics"][2]["value"]["share"]
    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "insert into evidently_metrics(timestamp, num_tickers, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s, %s)",
                (
                    current_date.to_pydatetime(),
                    num_tickers,
                    prediction_drift,
                    num_drifted_columns,
                    share_missing_values,
                ),
            )


def get_data(localrun: bool, env: str, fname: str) -> tuple:
    if localrun:
        configs = None
    else:
        configs = Configs(env)
    # load reference data
    ref_data, ref_params, ref_estimator = load_ref_data_model_params(
        configs, "ref_data_model"
    )

    # Use the ref params and estimator to prepare new data
    new_data = prepare_new_data(configs, env, fname, ref_params, ref_estimator)

    num_features = ref_data.columns.to_list()
    cat_features = ["Ticker"]
    for col in cat_features:
        num_features.remove(col)
    data_definition = DataDefinition(
        numerical_columns=num_features, categorical_columns=cat_features
    )

    return new_data, ref_data, data_definition


def batch_monitoring_backfill(
    localrun: bool, env: str, fname: str, backfill_horizon: int
) -> None:
    # get all data
    new_data, ref_data, data_definition = get_data(localrun, env, fname)
    # prepare the report
    report = Report(
        metrics=[
            ValueDrift(column="prediction"),
            DriftedColumnsCount(),
            MissingValueCount(column="prediction"),
        ]
    )

    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    for i in range(0, backfill_horizon):
        calculate_metrics_postgresql(i, new_data, ref_data, data_definition, report)

        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)
        logging.info(f"Day: {i + 1}/{backfill_horizon}: data sent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--localrun", action="store_true")
    parser.add_argument("--no-localrun", dest="localrun", action="store_false")
    parser.add_argument(
        "--env", type=str, required=True, help="env for the new data test, dev, prod"
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=True,
        help="filename for the new data, e.g.: Kaggle_Access_2025-07-28_WSPall_from_2020-07-28.parquet",
    )
    parser.add_argument("--backfill_horizon", type=int, required=False, default=20)
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    # batch_monitoring_backfill(
    #     localrun=True,
    #     env="dev",
    #     fname="Kaggle_Access_2025-07-28_WSPall_from_2020-07-28.parquet",
    #     backfill_horizon=20,
    # )
    batch_monitoring_backfill(
        localrun=args.localrun,
        env=args.env,
        fname=args.fname,
        backfill_horizon=args.backfill_horizon,
    )
