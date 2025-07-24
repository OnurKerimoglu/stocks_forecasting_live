import datetime
import json
import os

import mlflow
from mlflow.tracking import MlflowClient

# Global parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"

rootpath = os.path.dirname(os.path.dirname(__file__))
DEPLOYPATH = os.path.join(rootpath, "deployment")


def main_extract_model() -> None:
    run_id, params = retrieve_registered_model()
    store_model_artifacts(run_id, params)
    print("model parameters are extracted into: ", DEPLOYPATH)


def retrieve_registered_model() -> tuple:
    # Find the run_id, and extract the parameters
    mv = CLIENT.get_model_version_by_alias(name=REGISTRY_NAME, alias=MODEL_ALIAS)
    run_id = mv.run_id
    run = CLIENT.get_run(run_id)
    params = run.data.params
    # artifacts = CLIENT.list_artifacts(run_id, path=MODEL_ARTIFACT_FOLDER)
    return run_id, params


def store_model_artifacts(run_id: str, params: dict) -> None:
    # clean deployments_folder
    if os.path.exists(DEPLOYPATH):
        for f in os.listdir(DEPLOYPATH):
            try:
                os.remove(os.path.join(DEPLOYPATH, f))
            except Exception as e:
                print(e)
                os.rmdir(f)
    else:
        os.makedirs(DEPLOYPATH)
    # store params as json
    params_fpath = os.path.join(DEPLOYPATH, "params.json")
    with open(params_fpath, "w") as f:
        json.dump(params, f)
    print("params are stored in: ", params_fpath)

    # store requirements.txt and model.pkl
    artifact_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_FOLDER}"
    print(f"Downloading model and requirements from: {artifact_uri}")
    reqs_path_original = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/requirements.txt", dst_path=DEPLOYPATH
    )
    model_path_original = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/model.pkl", dst_path=DEPLOYPATH
    )
    reqs_path = os.path.join(DEPLOYPATH, "requirements.txt")
    # move the requirements.txt and model.pkl to the deployments folder
    os.rename(reqs_path_original, os.path.join(DEPLOYPATH, "requirements.txt"))
    os.rename(model_path_original, os.path.join(DEPLOYPATH, "model.pkl"))
    os.rmdir(os.path.join(DEPLOYPATH, MODEL_ARTIFACT_FOLDER))
    print("requirements.txt and model.pkl are stored in: ", reqs_path)

    # log date
    date_fpath = os.path.join(DEPLOYPATH, "extraction_date.txt")
    with open(date_fpath, "w") as f:
        f.write(datetime.datetime.now().isoformat())


if __name__ == "__main__":
    main_extract_model()
