import argparse
import json
import logging
import os
from pickle import dump

import pandas as pd
from tools import load_model_artifacts, prepare_data_for_monitoring

from scripts.gcp_functions import (
    upload_directory,
)
from scripts.load_configs import Configs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

TARGET = "returns"
ROOTPATH = os.path.dirname(os.path.dirname(__file__))
DATAPATH = os.path.join(ROOTPATH, "data")


def establish_baseline(localrun: bool, env: str, fname: str) -> None:
    configs = None if localrun else Configs(env)
    data, params, estimator = get_data(configs, env, fname)
    store_ref(
        configs,
        data,
        params,
        estimator,
        ref_path=os.path.join(DATAPATH, "ref_data_model"),
    )


def get_data(configs: dict, env: str, fname: str) -> tuple:
    localrun = False if configs else True

    params, estimator = load_model_artifacts(
        localrun,
        prefix="extracted_model",
        project=configs.cloud["gcs"]["project"] if configs else None,
        bucket=configs.cloud["gcs"]["models_bucket"] if configs else None,
        localrootdir=DATAPATH,
    )

    # Use the ref params and estimator to prepare new data
    data = prepare_data_for_monitoring(
        configs=configs,
        env=env,
        fname=fname,
        target=TARGET,
        params=params,
        estimator=estimator,
        localrootdir=DATAPATH,
    )

    return data, params, estimator


def store_ref(
    configs: dict,
    ref_data: pd.DataFrame,
    params: dict,
    estimator: object,
    ref_path: str,
) -> None:
    os.makedirs(ref_path, exist_ok=True)

    # save params as json
    with open(os.path.join(ref_path, "params.json"), "w") as f:
        json.dump(params, f)
    # save model pickle bin
    with open(os.path.join(ref_path, "model.pkl"), "wb") as f:
        dump(estimator, f)
    # save data as parquet
    ref_data.to_parquet(os.path.join(ref_path, "data.parquet"))

    logger.info(f"Reference data and model artifacts stored in: {ref_path}")

    if configs:
        upload_directory(
            project_id=configs.cloud["gcs"]["project"],
            bucket_name=configs.cloud["gcs"]["data_monitoring_bucket"],
            folder="ref_data_model",
            local_dir=ref_path,
        )
    else:
        logger.info("Localrun specified, skipping upload to cloud")


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
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    establish_baseline(localrun=args.localrun, env=args.env, fname=args.fname)
    # establish_baseline(
    #     localrun=True,
    #     env="prod",
    #     fname="Kaggle_Access_2025-07-22_WSPall_from_2020-07-22.parquet"
    # )
