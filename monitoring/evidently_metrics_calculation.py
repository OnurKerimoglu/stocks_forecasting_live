import datetime
import logging
import os
import random
import sys
import time

import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift

from data import (
    build_features,
    create_X_y_multistep,
)

rootpath = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(rootpath, "scripts"))
sys.path.append(os.path.join(rootpath, "src"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists evidently_metrics;
create table evidently_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""

CONNECTION_STRING = "host=localhost port=5432 user=postgres password=admin"
CONNECTION_STRING_DB = CONNECTION_STRING + " dbname=stocks"


def load_ref_data_model_params(configs: dict, prefix: str = "ref_data_model") -> tuple:
    from gcp_functions import load_json_from_gcs, load_pickle_from_gcs, read_file_as_df

    # Read ref_data, ref_params and ref_estimator from GCS
    ref_data = read_file_as_df(
        configs.cloud["gcs"]["project"],
        configs.cloud["gcs"]["data_monitoring_bucket"],
        f"{prefix}/data.parquet",
    )
    ref_data_flat = ref_data.reset_index(level="Ticker")
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
    return ref_data_flat, ref_params, ref_estimator


def prepare_new_data(data_path: dict, params: dict, estimator: object) -> pd.DataFrame:
    from gcp_functions import read_file_as_df

    df = read_file_as_df(
        data_path["project"],
        data_path["bucket"],
        data_path["prefix"] + "/" + data_path["fname"],
    )
    TARGET = "returns"
    df_feats, _features2scale = build_features(
        df, lags=int(params["lags"]), split="train", CldrFeats=params["CldrFeats"]
    )
    X, y = create_X_y_multistep(df_feats, steps=int(params["steps"]), target=TARGET)
    y_hat = estimator.predict(X)
    data = X.copy()
    data["target"] = y["y_step_1"]
    data["prediction"] = y_hat[:, 0]
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
    # begin is the last day available in data
    begin = new_data.index.max().to_pydatetime()
    current_data = new_data[
        (new_data.index >= (begin - datetime.timedelta(i)))
        & (new_data.index < (begin - datetime.timedelta(i - 1)))
    ]

    current_dataset = Dataset.from_pandas(current_data, data_definition=data_definition)
    reference_dataset = Dataset.from_pandas(ref_data, data_definition=data_definition)

    run = report.run(reference_data=reference_dataset, current_data=current_dataset)

    result = run.dict()

    prediction_drift = result["metrics"][0]["value"]
    num_drifted_columns = result["metrics"][1]["value"]["count"]
    share_missing_values = result["metrics"][2]["value"]["share"]
    with psycopg.connect(CONNECTION_STRING_DB, autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "insert into evidently_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
                (
                    begin + datetime.timedelta(i),
                    prediction_drift,
                    num_drifted_columns,
                    share_missing_values,
                ),
            )


def get_data(env: str, fname: str) -> tuple:
    from load_configs import Configs

    configs = Configs(env)
    # load reference data
    ref_data, ref_params, ref_estimator = load_ref_data_model_params(
        configs, "ref_data_model"
    )

    # Prepare new data, e.g., based on some new data in gcs dev bucket
    data_new_path = {
        "project": configs.cloud["gcs"]["project"],
        "bucket": configs.cloud["gcs"]["data_monitoring_bucket"],
        "prefix": f"cleaned_samples_{env}",
        "fname": fname,
    }

    # Let's use (we don't have to) the ref params and estimator to prepare new data
    new_data = prepare_new_data(data_new_path, ref_params, ref_estimator)

    num_features = ref_data.columns.to_list()
    categorical_columns = ["Ticker"]
    for col in categorical_columns:
        num_features.remove(col)
    data_definition = DataDefinition(
        numerical_columns=num_features, categorical_columns=categorical_columns
    )

    return new_data, ref_data, data_definition


def batch_monitoring_backfill(env: str, fname: str, backfill_horizon: int) -> None:
    # get all data
    new_data, ref_data, data_definition = get_data(env, fname)
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
        logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill(
        env="dev",
        fname="Kaggle_Access_2025-07-28_WSPall_from_2020-07-28.parquet",
        backfill_horizon=5,
    )
