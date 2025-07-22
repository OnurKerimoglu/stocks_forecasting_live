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
```
- Note that ruff is used as linter and formatter (will be installed as a dependency)

### Running the Training Pipeline
- **Workflow orchestration** is handled by [prefect](https://www.prefect.io/) (will be installed as a dependency)
- To start the prefect server: activate the .project venv on a terminal (see above), and issue `prefect server start` and on a browser, navigate to `http://localhost:8080` for the GUI.
- To create a work pool, deploy the training workflow, and start the workpool: on another terminal, after activating  the project venv, issue the following: 
```
prefect work-pool create --type process stocks_forecasting_live_local  # command-1
python deploy_training_workflow.py  # command-2
prefect worker start --pool "stocks_forecasting_live_local"  # command-3
```
where, in:
  - command-1: "stocks_forecasting_live_local" is the name of the pool (changing this may require other changes in the next two steps). 
  - command-2: will deploy the training workflow will be deployed from the git repository, from dev or prod branches, as set in the [deploy_training_worfklow.py](deploy_training_workflow.py). Note that without the next command it will be in 'Not Ready' status, as the work pool would not have been started.
  - command-3: the work pool that had been created by the command-1 will now be started, which should change the status of the deployment to 'Ready', as can be monitored from the GUI
- If the deploy_training_workflow.py is not changed before deployment, the workflow will be deployed in dev mode, and without schedules, otherwise with a weekly schedule. In any case, a manual run can be triggered, e.g., on the GUI, Deployments tab, 'Play' button on the top-right corner. Two parameters can be optionally set via 'Custom Run':
  - test_mode (default: True): raw_data will not be removed after execution
  - use_sample_tickers_for_training (default: True): Only two tickers (['AMZN', 'APPL']) will be used to train the model (these two tickers will be used for the model evaluation anyway, independent of the selection here)
- To stop the worker, and the prefect server, hit Ctrl+C in the respective terminals
- See [here](./documentation/documentation.md#workflow-orchestration) for the details on workflow orchestration.