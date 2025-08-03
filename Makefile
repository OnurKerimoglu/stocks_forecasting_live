# GCP settings
# should match with terraform/variables.tf and main.tf
PROJECT_ID   ?= stocks-forecasting-466906
REGION       ?= europe-west1
REPO         ?= stocks-forecasting-repo
IMAGE_NAME   ?= stocks_forecasting_inference
VERSION      ?= latest
SERVICE_NAME_ROOT ?= stocks-forecasting-service
SERVICE_ACCOUNT ?= stocks-forecasting-mle@$(PROJECT_ID).iam.gserviceaccount.com

GIT_TREE_STATE:=$(shell test -z "$$(git status --porcelain)" && echo clean || echo dirty)
BRANCH_ST:=$(shell echo $$(git rev-parse --abbrev-ref HEAD) | sed 's/\//_/g')
# if BRANCH_ST is not dev or prod, service name suffix will be test
ifeq ($(BRANCH_ST),dev)
  BRANCH_SIMPLE := dev
else ifeq ($(BRANCH_ST),prod)
  BRANCH_SIMPLE := prod
else
  BRANCH_SIMPLE := test
endif
IMAGE_URI:=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${VERSION}-${BRANCH_SIMPLE}
SERVICE_NAME:=${SERVICE_NAME_ROOT}-${BRANCH_SIMPLE}

# Targets
quality_checks:
	ruff check
	ruff format

.PHONY: tests
tests:
	pytest tests/

mlflow_serve:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --default-artifact-root ./artifacts

prefect_serve:
	prefect server start

prefect_create_workpool:
	prefect work-pool create --type process stocks_forecasting_live_local

prefect_deploy_train:
	python scripts/deploy_training_workflow.py --env ${BRANCH_SIMPLE}
	prefect worker start --pool "stocks_forecasting_live_local"

extract_registered_model:
	python scripts/extract_mlflow_artifacts.py --env ${BRANCH_SIMPLE} --cloudupload

inference_build_local: quality_checks tests
	@if [ "$(GIT_TREE_STATE)" = "dirty" ]; then \
	  echo "You have uncommitted changes in the repo. Please commit or stash them."; \
	else \
	  echo "Branch is clean, building…"; \
	  docker build -f Docker/Dockerfile -t ${IMAGE_URI} .; \
	fi

inference_serve_local:
	docker run -it --rm -p 9696:9696 ${IMAGE_URI}

inference_publish: inference_build_local
	@echo "Configuring Docker to auth with GAR"
	gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
	@echo "Pushing Docker image to GAR"
	docker push $(IMAGE_URI)
	@echo "Image published: $(IMAGE_URI)"

inference_deploy: inference_publish
	@echo "Deploying $(SERVICE_NAME) to Cloud Run"
	gcloud run deploy $(SERVICE_NAME) \
	  --image=$(IMAGE_URI) \
	  --region=$(REGION) \
	  --platform=managed \
	  --project=$(PROJECT_ID) \
	  --service-account=$(SERVICE_ACCOUNT) \
	  --allow-unauthenticated \
	  --port=9696

inference_test_raw:
	@echo "Testing Cloud Run service"
	@SERVICE_URL="$$(gcloud run services describe $(SERVICE_NAME) \
	  --project=$(PROJECT_ID) \
	  --region=$(REGION) \
	  --format='value(status.url)')" && \
	echo "Service URL of the deployment is: $$SERVICE_URL" && \
	curl -X POST "$$SERVICE_URL/forecast" \
	  -H "Content-Type: application/json" \
	  -d '{"ticker":"GOOG", "past_horizon": 5}'

inference_test_pretty:
	python scripts/test_inference.py --env ${BRANCH_SIMPLE} --ticker GOOG --past_horizon 5

FNAME_NEW="Kaggle_Access_2025-07-22_WSPall_from_2020-07-22.parquet"
ENV_NEW="prod"
monitoring_establish_baseline:
	python monitoring/establish_baseline.py --no-localrun --env ${ENV_NEW} --fname ${FNAME_NEW}

FNAME_NEW="Kaggle_Access_2025-07-28_WSPall_from_2020-07-28.parquet"
ENV_NEW="dev"
BACKFILL_HORIZON=20
monitoring_base_refresh:
	python monitoring/refresh_monitoring_dashboard.py --no-localrun --env ${ENV_NEW} --fname ${FNAME_NEW} --backfill_horizon ${BACKFILL_HORIZON}
