import argparse
import json
import os

from gcp_functions import download_directory
from load_configs import Configs
from utils import resolve_model_bundle_uri_for_env

# Global parameters
ROOTPATH = os.path.dirname(os.path.dirname(__file__))
MODELPATH = os.path.join(ROOTPATH, "models")
EXTRACTED_MODEL_DIRNAME = "extracted_model"


def download_extracted_model_from_gcs(env: str, refresh: bool = True) -> None:
    local_dir = os.path.join(MODELPATH, EXTRACTED_MODEL_DIRNAME)
    config_cloud = Configs(env).cloud
    gcp_project = config_cloud["gcs"]["project"]
    gcp_bucket = config_cloud["gcs"]["mlflow_bucket"]
    bundle_uri, manifest = resolve_model_bundle_uri_for_env(
        env=env,
        gcp_project=gcp_project,
        gcp_bucket=gcp_bucket,
        status_blob="promotion_status.json",
        local_status_path=os.path.join(MODELPATH, "promotion_status.json"),
    )
    dirname = bundle_uri.split(gcp_bucket + "/")[1]
    download_directory(
        project_id=gcp_project,
        bucket_name=gcp_bucket,
        folder=dirname,
        local_dir=local_dir,
        refresh=refresh,
    )
    with open(os.path.join(local_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="test, dev, prod")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-refresh", dest="refresh", action="store_false")
    parser.set_defaults(feature=True)
    # args = parser.parse_args()
    # download_extracted_model_from_gcs(env=args.env, refresh=args.refresh)
    download_extracted_model_from_gcs(env="dev", refresh=True)
