import datetime
import json
import os
import pickle

import mlflow
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor

# Global parameters
mlflow.set_tracking_uri("http://127.0.0.1:5000")
CLIENT = MlflowClient()
MODEL_ARTIFACT_FOLDER = "mlflow_models"
REGISTRY_NAME = "stocks_forecasting_candidates"  # The registry from which the models should be pulled from
MODEL_ALIAS = "champion"

rootpath = os.path.dirname(os.path.dirname(__file__))
DEPLOYPATH = os.path.join(rootpath, "deployment")


def main_extract_model() -> None:
    model, params, run_id = retrieve_registered_model()
    store_model_artifacts(model, params, run_id)
    print("model parameters are extracted into: ", DEPLOYPATH)


def retrieve_registered_model() -> tuple:
    # Load the model from the Model Registry
    model_uri = f"models:/{REGISTRY_NAME}@{MODEL_ALIAS}"
    print(f"Retrieveing model_uri: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    # Get the parameters
    mv = CLIENT.get_model_version_by_alias(name=REGISTRY_NAME, alias=MODEL_ALIAS)
    run = CLIENT.get_run(mv.run_id)
    params = run.data.params
    # artifacts = CLIENT.list_artifacts(mv.run_id, path=MODEL_ARTIFACT_FOLDER)
    return model, params, mv.run_id


def store_model_artifacts(model: XGBRegressor, params: dict, run_id: str) -> None:
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
    # store model as pkl
    model_fpath = os.path.join(DEPLOYPATH, "model.pkl")
    with open(model_fpath, "wb") as f:
        pickle.dump(model, f)
    print("model is stored in: ", model_fpath)
    # store params as json
    params_fpath = os.path.join(DEPLOYPATH, "params.json")
    with open(params_fpath, "w") as f:
        json.dump(params, f)
    print("params are stored in: ", params_fpath)
    # store requirements.txt
    artifact_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_FOLDER}/requirements.txt"
    print(f"Downloading requirements from: {artifact_uri}")
    reqs_path_original = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path=DEPLOYPATH
    )
    reqs_path = os.path.join(DEPLOYPATH, "requirements.txt")
    # move the requirements.txt to the deployments folder
    os.rename(reqs_path_original, reqs_path)
    os.rmdir(os.path.join(DEPLOYPATH, MODEL_ARTIFACT_FOLDER))
    print("requirements are stored in: ", reqs_path)
    # log date
    date_fpath = os.path.join(DEPLOYPATH, "extraction_date.txt")
    with open(date_fpath, "w") as f:
        f.write(datetime.datetime.now().isoformat())
    return DEPLOYPATH


if __name__ == "__main__":
    main_extract_model()
