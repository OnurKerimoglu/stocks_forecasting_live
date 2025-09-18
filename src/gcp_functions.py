import io
import json
import os
import pickle
import subprocess

import pandas as pd
from google.cloud import storage


def read_file_as_df(project_id: str, bucket_name: str, gcs_path: str) -> None:
    # Initialize client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # Download to bytes and read
    file_bytes = blob.download_as_bytes()
    extension = gcs_path.split(".")[-1]
    if extension == "parquet":
        df = pd.read_parquet(io.BytesIO(file_bytes))
    return df


def load_pickle_from_gcs(project_id: str, bucket_name: str, gcs_path: str) -> None:
    # Initialize client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # load from pickle
    if blob.exists:
        with blob.open(mode="rb") as f:
            loaded_obj = pickle.load(f)
    else:
        loaded_obj = None
    return loaded_obj


def load_json_from_gcs(project_id: str, bucket_name: str, gcs_path: str) -> None:
    # Initialize client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    # check if the blob exists
    if blob.exists():
        dict = json.loads(blob.download_as_string(client=None))
    else:
        dict = None
    return dict


def clear_gcs_folder(project_id: str, bucket_name: str, folder: str) -> None:
    """
    Deletes all objects in a GCS bucket under the specified folder.
    Parameters:
        project_id (str): The GCP project ID.
        bucket_name (str): The name of the GCS bucket.
        folder (str): folder (prefix) to clear.
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder))
    if blobs:
        print(
            f"Clearing {len(blobs)} existing object(s) under gs://{bucket.name}/{folder}"
        )
        bucket.delete_blobs(blobs)
        print(f"Successfully cleared gs://{bucket.name}/{folder}")


def upload_directory(
    project_id: str, bucket_name: str, folder: str, local_dir: str
) -> None:
    """
    Uploads all files from a local directory to a GCS bucket under the specified prefix.

    Parameters:
        project_id (str): The GCP project ID
        bucket_name (str): The name of the GCS bucket
        folder (str): GCS target folder (prefix, relative to the bucket)
        local_dir (str): Local directory to upload
        bucket (storage.Bucket): Target GCS bucket
    """
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            if folder is not None:
                blob_path = os.path.join(folder, rel_path).replace("\\", "/")
            else:
                blob_path = rel_path

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket.name}/{blob_path}")

    print(f"Successfully uploaded {len(files)} file(s) to gs://{bucket.name}/{folder}")


def upload_file_to_folder(
    project_id: str, bucket_name: str, folder: str, file: str
) -> None:
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    fname = os.path.basename(file)
    # construct blob name with fname and prefix = folder
    blob_name = f"{folder}/{fname}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file)
    print(f"Uploaded {file} to gs://{bucket.name}/{folder}")


def download_directory(
    project_id: str,
    bucket_name: str,
    folder: str,
    local_dir: str,
    refresh: bool = False,
) -> None:
    """
    Downloads all files under a GCS 'folder' (prefix) to a local directory.

    Parameters:
        project (str, optional): GCP project ID (optional if inferred from credentials).
        bucket_name (str): Name of the GCS bucket.
        gcs_prefix (str): The prefix (folder path) in GCS to download from.
        local_dir (str): Local directory where files should be saved.
    """
    # Ensure local output directory exists
    # clean the local folder
    if os.path.exists(local_dir):
        if refresh:
            print(f"Clearing {local_dir}")
            for f in os.listdir(local_dir):
                try:
                    os.remove(os.path.join(local_dir, f))
                except Exception as e:
                    print(e)
                    os.rmdir(f)
        else:
            print("Specified directory exists and refresh is set to False. Exiting.")
            return
    else:
        os.makedirs(local_dir)

    # Initialize client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=folder))

    if not blobs:
        print(f"No files found under gs://{bucket_name}/{folder}")
        return

    print(f"Found {len(blobs)} file(s) under gs://{bucket_name}/{folder}")

    for blob in blobs:
        if blob.name.endswith("/"):  # Skip directory placeholders
            continue

        rel_path = os.path.relpath(blob.name, folder)
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        blob.download_to_filename(local_path)
        print(f"Downloaded gs://{bucket_name}/{blob.name} → {local_path}")

    print(f"Successfully downloaded {len(blobs)} file(s) to {local_dir}")


def get_gcrun_service_url(service_name: str, region: str, project_id: str) -> str:
    cmd = [
        "gcloud",
        "run",
        "services",
        "describe",
        service_name,
        "--project",
        project_id,
        "--region",
        region,
        "--format=value(status.url)",
    ]
    url = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    return url
