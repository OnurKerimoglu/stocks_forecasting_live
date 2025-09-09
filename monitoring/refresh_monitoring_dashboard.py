import argparse
import datetime
import logging
import os
import random
import time
import warnings

import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, MissingValueCount, ValueDrift
from tools import load_model_artifacts, prepare_data_for_monitoring

from load_configs import Configs
from raw_data import load_data

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

TARGET = "returns"
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
    logger.info(f"Processing: {current_date}")
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
    configs = None if localrun else Configs(env)
    # load reference data
    ref_data = load_data(
        localrun,
        prefix="ref_data_model",
        fname="data.parquet",
        project=configs.cloud["gcs"]["project"] if configs else None,
        bucket=configs.cloud["gcs"]["data_monitoring_bucket"] if configs else None,
        localrootdir=DATAPATH,
    )

    ref_params, ref_estimator = load_model_artifacts(
        localrun,
        prefix="ref_data_model",
        project=configs.cloud["gcs"]["project"] if configs else None,
        bucket=configs.cloud["gcs"]["data_monitoring_bucket"] if configs else None,
        localrootdir=DATAPATH,
    )

    # Use the ref params and estimator to prepare new data
    new_data = prepare_data_for_monitoring(
        configs, env, fname, TARGET, ref_params, ref_estimator, localrootdir=DATAPATH
    )

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
        logger.info(f"Day: {i + 1}/{backfill_horizon}: data sent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--localrun", action="store_true")
    parser.add_argument("--no-localrun", dest="localrun", action="store_false")
    parser.add_argument(
        "--env", type=str, required=False, help="env for the new data test, dev, prod"
    )
    parser.add_argument(
        "--fname",
        type=str,
        required=False,
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
