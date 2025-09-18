import argparse
import json
import os

from gcp_functions import download_directory, load_json_from_gcs
from load_configs import Configs


def _load_local_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_promotion_status(
    *,
    gcp_project: str,
    gcp_bucket: str,
    status_blob: str = "promotion_status.json",
    local_status_path: str = "./models/promotion_status.json",
) -> dict:
    """
    Load promotion_status.json from GCS if available; fall back to local file.
    Raises FileNotFoundError if neither is available, or ValueError for malformed content.
    """

    # Try GCS if config is provided and helper is available
    if gcp_project and gcp_bucket and load_json_from_gcs is not None:
        try:
            status = load_json_from_gcs(gcp_project, gcp_bucket, status_blob)
        except Exception:
            status = None  # ignore and fall back

    # Fallback to local
    if status is None and os.path.exists(local_status_path):
        status = _load_local_json(local_status_path)

    if status is None:
        raise FileNotFoundError(
            f"promotion_status.json not found in cloud (gs://{gcp_bucket}/{status_blob}) "
            f"or locally ({local_status_path})."
        )
    if not isinstance(status, dict):
        raise ValueError("promotion_status.json content is not a JSON object.")
    else:
        print(
            f"Loaded promotion status: {status} (last updated: {status['last_updated']})"
        )
    return status


def resolve_bundle_uri_for_env(
    env: str,
    *,
    gcp_project: str,
    gcp_bucket: str,
    status_blob: str = "promotion_status.json",
    local_status_path: str = "./models/promotion_status.json",
) -> str:
    """
    Return the bundle_uri for the given environment by reading promotion_status.
    Mapping: dev -> 'challenger', prod -> 'champion'
    """
    status = load_promotion_status(
        gcp_project=gcp_project,
        gcp_bucket=gcp_bucket,
        status_blob=status_blob,
        local_status_path=local_status_path,
    )

    env_norm = (env or "").strip().lower()
    if env_norm in {"dev", "development"}:
        key = "challenger"
    elif env_norm in {"prod", "production"}:
        key = "champion"
    else:
        raise ValueError(f"Unsupported env '{env}'. Expected 'dev' or 'prod'.")

    print(f"In promotion status, referring to key: '{key}' for env '{env}'.")

    ptr = status.get(key)
    if not isinstance(ptr, dict):
        raise RuntimeError(
            f"Key '{key}' missing or not an object in promotion_status.json."
        )

    bundle_uri = ptr.get("bundle_uri")
    if not bundle_uri:
        raise RuntimeError(
            f"'bundle_uri' missing under '{key}' in promotion_status.json."
        )

    manifest = {"alias": key, "model_ref": ptr}
    return bundle_uri, manifest


def download_extracted_model_from_gcs(env: str, refresh: bool = True) -> None:
    rootpath = os.path.dirname(os.path.dirname(__file__))
    modelspath = os.path.join(rootpath, "models")
    local_dir = os.path.join(modelspath, "extracted_model")
    config_cloud = Configs(env).cloud
    gcp_project = config_cloud["gcs"]["project"]
    gcp_bucket = config_cloud["gcs"]["mlflow_bucket"]
    bundle_uri, manifest = resolve_bundle_uri_for_env(
        env=env,
        gcp_project=gcp_project,
        gcp_bucket=gcp_bucket,
        status_blob="promotion_status.json",
        local_status_path=os.path.join(modelspath, "promotion_status.json"),
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
    args = parser.parse_args()
    download_extracted_model_from_gcs(env=args.env, refresh=args.refresh)
    # download_extracted_model_from_gcs(env="prod", refresh=True)
