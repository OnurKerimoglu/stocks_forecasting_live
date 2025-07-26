import os

from google.cloud import storage


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
            blob_path = os.path.join(folder, rel_path).replace("\\", "/")

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded {local_path} to gs://{bucket.name}/{blob_path}")

    print(f"Successfully uploaded {len(files)} file(s) to gs://{bucket.name}/{folder}")
