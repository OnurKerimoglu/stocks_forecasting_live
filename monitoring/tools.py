import logging
import os
from json import load as jload
from pickle import load as pload

import pandas as pd

from data import build_features, create_X_y_multistep
from gcp_functions import (
    load_json_from_gcs,
    load_pickle_from_gcs,
)
from raw_data import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


def load_model_artifacts(
    localrun: bool, prefix: str, project: str, bucket: str, localrootdir: str
) -> tuple:
    if not localrun:
        logging.info(f"Loading model artifacts from GCS bucket: {bucket}/{prefix}")
        # Read ref_params and ref_estimator from GCS
        ref_params = load_json_from_gcs(
            project,
            bucket,
            f"{prefix}/params.json",
        )
        ref_estimator = load_pickle_from_gcs(
            project,
            bucket,
            f"{prefix}/model.pkl",
        )
    else:
        # Read from the local filesystem
        ref_path = os.path.join(localrootdir, prefix)
        if not os.path.exists(ref_path):
            raise Exception(
                f"no config provided for GCS and local path {ref_path} does not exist"
            )
        else:
            logger.info(f"Loading model artifacts from filesystem: {ref_path}")
            with open(os.path.join(ref_path, "params.json")) as f:
                ref_params = jload(f)
            with open(os.path.join(ref_path, "model.pkl"), "rb") as f:
                ref_estimator = pload(f)
    return ref_params, ref_estimator


def prepare_data_for_monitoring(
    configs: dict,
    env: str,
    fname: str,
    target: str,
    params: dict,
    estimator: object,
    localrootdir: str | None = None,
) -> pd.DataFrame:
    # load the new data
    localrun = False if configs else True
    df = load_data(
        localrun,
        prefix=f"cleaned_samples_{env}",
        fname=fname,
        project=configs.cloud["gcs"]["project"] if configs else None,
        bucket=configs.cloud["gcs"]["data_monitoring_bucket"] if configs else None,
        localrootdir=localrootdir,
    )
    logger.info("Creating data with features, target and predictions")
    CldrFeats = params["CldrFeats"] if "CldrFeats" in params.keys() else "True"
    CldrFeats = True if CldrFeats == "True" else False
    df_feats, _features2scale = build_features(
        df, lags=int(params["lags"]), CldrFeats=CldrFeats
    )
    X, y = create_X_y_multistep(df_feats, steps=int(params["steps"]), target=target)
    y_hat = estimator.predict(X)
    data = X.copy()
    data["target"] = y["y_step_1"]
    data["prediction"] = y_hat[:, 0]
    # Convert multi-index df to single index
    data_flat = data.reset_index(level="Ticker")
    return data_flat
