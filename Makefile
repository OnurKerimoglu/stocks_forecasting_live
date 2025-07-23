# Set default arguments
train_deployment_mode ?= dev

# Targets
quality_checks:
	ruff check
	ruff format

prefect_serve:
	prefect server start

prefect_create_workpool:
	prefect work-pool create --type process stocks_forecasting_live_local

prefect_deploy_train:
	python deploy_training_workflow.py --mode ${train_deployment_mode}
	prefect worker start --pool "stocks_forecasting_live_local"

mlflow_serve:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
