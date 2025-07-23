quality_checks:
	ruff check
	ruff format

prefect_serve:
	prefect server start

prefect_create_workpool:
	prefect work-pool create --type process stocks_forecasting_live_local

prefect_deploy_train:
	python deploy_training_workflow.py
	prefect worker start --pool "stocks_forecasting_live_local"
