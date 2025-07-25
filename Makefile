# GCP settings
# should match with terraform/variables.tf and main.tf
PROJECT_ID   ?= stocks-forecasting-466906
REGION       ?= europe-west1
REPO         ?= stocks-forecasting-repo
IMAGE_NAME   ?= stocks_forecasting_inference
VERSION      ?= latest

IMAGE_URI:=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${VERSION}

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
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --default-artifact-root ./artifacts

extract_registered_model:
	python src/extract_mlflow_artifacts.py

inference_build_local:
	docker build -f Docker/Dockerfile -t ${IMAGE_URI} .

inference_serve_local:
	docker run -it --rm -p 9696:9696 ${IMAGE_URI}

inference_test_local:
	python scripts/test_inference.py --ticker GOOG

inference_publish: inference_build
	@echo "Configuring Docker to auth with GAR"
	gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
	@echo "Pushing Docker image to GAR"
	docker push $(IMAGE_URI)
	@echo "Image published: $(IMAGE_URI)"
