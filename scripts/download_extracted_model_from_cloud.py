import argparse
import os

from src.gcp_functions import download_directory
from src.load_configs import Configs


def download_extracted_model_from_gcs(refresh: bool = True) -> None:
    rootpath = os.path.dirname(os.path.dirname(__file__))
    local_dir = os.path.join(rootpath, "extracted_model")
    config_cloud = Configs().cloud
    download_directory(
        project_id=config_cloud["gcs"]["project"],
        bucket_name=config_cloud["gcs"]["models_bucket"],
        folder="extracted_model",
        local_dir=local_dir,
        refresh=refresh,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-refresh", dest="refresh", action="store_false")
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    download_extracted_model_from_gcs(refresh=args.refresh)
