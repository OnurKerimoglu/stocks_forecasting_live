import json
import logging
import os

import tomli

from gcp_functions import load_json_from_gcs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_local_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_promotion_status(
    *,
    gcp_project: str,
    gcp_bucket: str,
    status_blob: str,
    local_status_path: str,
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


def resolve_model_bundle_uri_for_env(
    env: str,
    *,
    gcp_project: str,
    gcp_bucket: str,
    status_blob: str = "promotion_status.json",
    local_status_path: str = "./models/promotion_status.json",
) -> str:
    """
    Return the modelbundle_uri for the given environment by reading promotion_status.
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


def get_pipreqs_from_pyproject(path: str) -> list[str]:
    """
    Parses your dependencies from pyproject.toml and returns them as a list
    of pip-style requirement strings.
    """
    with open(path, "rb") as f:
        pyproject = tomli.load(f)

    deps = pyproject.get("project", {}).get("dependencies", [])

    return deps


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )
    rootpath = os.path.dirname(os.path.dirname(__file__))
    pipreqs = get_pipreqs_from_pyproject(os.path.join(rootpath, "pyproject.toml"))
    print(pipreqs)
