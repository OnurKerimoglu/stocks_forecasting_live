# Stocks Forecasting

## Motivation and Objectives

## Instructions for Reproduction
### Prerequisites and Initial Setup
- The [training dataset](https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating/data) will be downloaded with Kaggle API, which requires signing up and creating an API token (see: https://www.kaggle.com/docs/api))
- The installation instructions below assumes availability of uv on your system (https://docs.astral.sh/uv/getting-started/installation/)
- On a terminal, run:
```
git clone https://github.com/OnurKerimoglu/stocks_forecasting_live.git   # clone the project repo
cd stocks_forecasting_live                  # cd into project repo
uv venv  # creates a virtual environemnt    # Create a venv
source .venv/bin/activate                   # Activate the venv
uv sync                                     # Install the dependencies as specified in pyproject.toml and uv.lock
uv pip install -e .                         # Install the project as a package (`uv pip install -e .`)
pre-commit install                          # Enable pre-commit hooks
```
Notes:
- A [Makefile](Makefile) automates/facilitates certain operations, as will be referred below.
- [ruff](https://docs.astral.sh/ruff/) is used as linter and formatter. Issue `make quality_checks` to run the quality checks manually.
- [pytest](https://docs.pytest.org) is used as the testing framework. Issue `make tests` to run the (so far only unit) tests manually.
- [pre-commit](https://pre-commit.com/) hooks will be enabled by the last command above. Only the default hooks, private key detection and linting/formatting hooks are specified (see the [config](.pre-commit-config.yaml))

### Cloud Infrastructure

This project uses certain GCP services for deployment, managed by [Hashicorp Terraform](https://developer.hashicorp.com/terraform) (see: [terraform/main.tf](terraform/main.tf)). For being able to reproduce the steps that depend on cloud resources:
- Create a project on GCP
- Create a service account with the following roles:
  - Artifact Registry Administrator
  - Cloud Functions Admin
  - Service Account User
  - Service Usage Admin
- Create a key for the service account
- Create a file named 'terraform.tfvars' in the terraform directory that contains the path to your service account key
- In the terraform directory:
  - `terraform init`: to initilize the backend and provider (google) plugins
  - `terraform apply`, and when prompted confirm: to generate the resources


### Training Pipeline

#### Initiating the Orchestrator
is handled by [prefect](https://www.prefect.io/) (will be installed as a dependency). The workflow deployment comprise the following steps:
- Start the prefect server: activate the .project venv on a terminal (see above), and issue `make prefect_serve` (see the [Makefile](Makefile)). Then on a browser, navigate to `http://localhost:8080` to access the GUI.
- Create a work pool: once the prefect server is running (previous step), open a new terminal, after activating the project venv, issue: `make prefect_create_workpool`. Note that this step is required only once, and will actually return an error if repeated
- Deploy the training workflow and start the worker: once the server has started and a worker pool is created (previous 2 steps), issue `make prefect_deploy_train train_deployment_mode=dev`. Note that this command will call the [deploy_training_worfklow.py](deploy_training_workflow.py) script, which in turn will deploy the training workflow from the local filesystem as train_deployment_mode=dev in the make command (if train_deployment_mode=prod, the deployment will be made from (this) git repository, 'prod' branch. The deployed workflow can be seen in the GUI, Deployments tab, with 'Ready' status, once the worker has started (may take a few seconds).

To stop the worker, and the prefect server, hit Ctrl+C in the respective terminals. See [here](./documentation/documentation.md#workflow-orchestration) for the details on workflow orchestration.

#### Experiment Tracking and Model Registry
is handled by [mlflow](https://mlflow.org/). To activate mlflow server, simply open a new terminal, activate the venv and issue `make mlflow_serve` (see the [Makefile](Makefile)). On a browser, navigate to `http://localhost:5000` to access the Mlflow GUI. To stop the server, hit Ctrl+C on the terminal.  See [here](./documentation/documentation.md#experiment-tracking-and-model-registry) for the details on experiment tracking and model registry.

#### Manually Triggering an Experiment on the Training Pipeline

If the deploy_training_workflow.py is not changed before deployment, the workflow will be deployed in dev mode, and without schedules, otherwise with a weekly schedule. In any case, a manual run can be triggered, e.g., on the GUI, Deployments tab, 'Play' button on the top-right corner. Three parameters can be optionally set via 'Custom Run':
  - test_mode (default: True): raw_data will not be removed after execution
  - use_sample_tickers_for_training (default: True): Only two tickers (['AMZN', 'APPL']) will be used to train the model (these two tickers will be used for the model evaluation anyway, independent of the selection here)
  - select_only_latest (default: True): if True, the best model run will be selected only among runs from the current date, i.e., ignoring the previous runs

### Inference Pipeline

#### Local Build
Deployment of the inference pipeline is a two-step process:
1. Model extraction from mlflow: issue `make extract_registered_model`, only after making sure that the mlflow server is running (if not `make mlflow_serve`). This will query mlflow and get the run_id of the model registered with alias 'champion' (i.e., last version), and copy the `model.pkl` and `requirements.txt` artifacts as well as the parameters as `params.json` into a `deployment` folder under project root (after removing its previous contents).
2. Building the container image:  issue `make inference_build_local`. After triggering the `quality_checks` and `tests` targets (see [initial setup](#prerequisites-and-initial-setup)) to catch any obvious flaws, this will pack all necessary files and install packages needed for serving the inference pipeline.

#### Local Testing
To test the inference pipeline and try some forecasts:
1. Start serving the flask app: issue `make inference_serve_local`. This will start the flask app at `http://0.0.0.0:9696`
2. Run some tests: once the inference is serving, issue `make inference_test_local` to run a test. This will call a [python script](scripts/test_inference.py) (with a ticker symbol as argument), which will in turn send a post request to localhost:9696/forecast, parse the output and -if everything worked- return something like:
```
=== LAST DAY ===
             close returns (%)
index
2025-07-24  194.69       1.66%

=== FORECAST ===
             close returns (%)
index
2025-07-25  195.61       0.47%
2025-07-28  195.55      -0.03%
2025-07-29  194.63      -0.48%
2025-07-30  195.33       0.36%
2025-07-31  193.93      -0.72%
```

#### Deploying the image to Cloud Run and Testing
Assuming that the [Cloud Infrastructure](#cloud-infrastructure) instructions have been successfully executed, three steps are needed for deployment:
1. Build the image locally (see above)
2. Publish the image to [Google Artifact Registery](https://cloud.google.com/artifact-registry/docs) (GAR): issue `make inference_publish`
3. Deploy the image in GAR to [Cloud Run](https://cloud.google.com/run?hl=en): issue `make inference_deploy`

Note that, as the second step is a dependency of the third, and the first is a depednency of first, issuing directly the third will suffice.

Once the deployment is done, a Service URL will be displayed. Note that this is a revision-specific URL that will change with every deployment. `cloud run services describe` with the correct parameters can provide the stable URL. the Makefile target `inference_test_deployment` makes use of this function and constructs a curl command for a default ticker to test the deployment.
