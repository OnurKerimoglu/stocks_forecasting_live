import argparse
import datetime
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

from gcp_functions import clear_gcs_folder, upload_directory
from load_configs import Configs
from mlflow_helpers import MLFlowRetriever

# Global parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"

rootpath = os.path.dirname(os.path.dirname(__file__))
LOCALPATH = os.path.join(rootpath, "data", "extracted_model")


def main_extract_model(env: str, cloudupload: bool = True) -> None:
    # collect and store additional metadata
    run_id, metadata = MLFlowRetriever(client=CLIENT).retrieve_mlflow_model_metadata(
        registry_name=REGISTRY_NAME, model_alias=MODEL_ALIAS
    )
    params = metadata["params"]
    store_model_artifacts_local(run_id, params, metadata)
    print("model parameters are extracted into: ", LOCALPATH)
    if cloudupload:
        configs = Configs(env)
        upload_to_gcs(configs.cloud, localpath=LOCALPATH)
    else:
        print("model artifacts are not uploaded to GCS")


def store_model_artifacts_local(run_id: str, params: dict, metadata: dict) -> None:
    # clean the local folder
    if os.path.exists(LOCALPATH):
        for f in os.listdir(LOCALPATH):
            try:
                os.remove(os.path.join(LOCALPATH, f))
            except Exception as e:
                print(e)
                os.rmdir(f)
    else:
        os.makedirs(LOCALPATH)
    # store params as json
    params_fpath = os.path.join(LOCALPATH, "params.json")
    with open(params_fpath, "w") as f:
        json.dump(params, f)
    print("params stored in: ", params_fpath)

    # store metadata as json
    meta_fpath = os.path.join(LOCALPATH, "metadata.json")
    with open(meta_fpath, "w") as f:
        json.dump(metadata, f)
    print("metadata stored in: ", meta_fpath)

    # store requirements.txt and model.pkl
    artifact_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_FOLDER}"
    print(f"Downloading model and requirements from: {artifact_uri}")
    reqs_path_original = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/requirements.txt", dst_path=LOCALPATH
    )
    model_path_original = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/model.pkl", dst_path=LOCALPATH
    )
    reqs_path = os.path.join(LOCALPATH, "requirements.txt")
    # move the requirements.txt and model.pkl to the local folder
    os.rename(reqs_path_original, os.path.join(LOCALPATH, "requirements.txt"))
    os.rename(model_path_original, os.path.join(LOCALPATH, "model.pkl"))
    os.rmdir(os.path.join(LOCALPATH, MODEL_ARTIFACT_FOLDER))
    print("requirements.txt and model.pkl are stored in: ", reqs_path)

    # log date
    date_fpath = os.path.join(LOCALPATH, "extraction_date.txt")
    with open(date_fpath, "w") as f:
        f.write(datetime.datetime.now().isoformat())


def upload_to_gcs(configs: dict, localpath: str) -> None:
    """
    Syncing a local directory to a GCS Bucket.

    Parameters:
        localpath (str): Path to the local directory.
        configs (dict): Configuration dict with GCS project and bucket.
    """
    if not os.path.isdir(localpath):
        raise FileNotFoundError(f"Local directory '{localpath}' does not exist.")

    project_id = configs["gcs"]["project"]
    bucket_name = configs["gcs"]["models_bucket"]
    gcs_folder = os.path.basename(localpath.rstrip("/")) + "/"

    clear_gcs_folder(project_id, bucket_name, gcs_folder)
    upload_directory(project_id, bucket_name, gcs_folder, localpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="test, dev, prod")
    parser.add_argument("--cloudupload", action="store_true")
    parser.add_argument("--no-cloudupload", dest="cloud_upload", action="store_false")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main_extract_model(env=args.env, cloudupload=args.cloudupload)
    # main_extract_model(env="test", cloudupload=False)
