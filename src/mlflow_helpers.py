from mlflow.tracking import MlflowClient


class MLFlowRetriever:
    def __init__(self, client: MlflowClient) -> None:
        self.client = client

    def retrieve_mlflow_model_metadata(
        self,
        registry_name: str | None = None,
        model_alias: str | None = None,
    ) -> tuple:
        # Find the run_id, and extract the parameters
        mv = self.client.get_model_version_by_alias(
            name=registry_name, alias=model_alias
        )
        run_id = mv.run_id
        run = self.client.get_run(run_id)
        metadata = self.extract_metadata(
            run_id, run.info.__dict__, run.data.tags, run.data.metrics
        )
        metadata["aliases"] = "-".join(mv.aliases)
        metadata["version"] = mv.version
        metadata["params"] = run.data.params
        metadata["registry_name"] = registry_name
        metadata["model_alias"] = model_alias
        return run_id, metadata

    def extract_metadata(
        self, run_id: str, run_info: dict, run_tags: dict, run_metrics: dict
    ) -> dict:
        meta = {}
        meta["run_id"] = run_id
        meta["run_info"] = run_info
        meta["tags"] = run_tags
        meta["metrics"] = run_metrics
        return meta
