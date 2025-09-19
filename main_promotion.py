import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from gcp_functions import load_json_from_gcs, upload_file_to_folder
from load_configs import Configs


@dataclass
class ModelRef:
    run_id: str
    model_version: str | None
    bundle_uri: str
    alias: str | None = None
    metrics: dict[str, Any] | None = None
    last_updated: str | None = None


@dataclass
class PromotionStatus:
    last_updated: str
    challenger: ModelRef | None
    champion: ModelRef | None
    archived: dict[list]  # {"challenger":[...], "champion":[...]}

    def to_dict(self) -> dict[str, Any]:
        def ptr(m: ModelRef | None) -> dict | None:
            if not m:
                return None
            return {
                "run_id": m.run_id,
                "model_version": m.model_version,
                "bundle_uri": m.bundle_uri,
                "last_updated": m.last_updated,
            }

        return {
            "last_updated": self.last_updated,
            "challenger": ptr(self.challenger),
            "champion": ptr(self.champion),
            "archived": self.archived,
        }


class ModelPromoter:
    """
    Manages promotions and maintains a single promotion_status.json (local + GCS).

    File schema:
    {
        "last_updated": "<ISO8601 Europe/Berlin>",
        "challenger": { "run_id": "...", "model_version": "...", "bundle_uri": "...", "last_updated": "..." } | null,
        "champion":   { "run_id": "...", "model_version": "...", "bundle_uri": "...", "last_updated": "..." } | null,
        "archived": {
            "challenger": [ {"run_id": "...", "model_version": "...", "when": "..."} ],
            "champion":   [ {"run_id": "...", "model_version": "...", "when": "..."} ]
        }
    }
    """

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
    ) -> None:
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
        )

        # Input Args
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Constants
        self.status_blob = "promotion_status.json"
        self.registry_name = "stocks_forecasting"
        self.candidate_alias = "Candidate"
        self.challenger_alias = "Challenger"
        self.champion_alias = "Champion"
        self.model_artifact_folder = "mlflow_models"

        # Derived Constants
        self.now = datetime.now(tz=UTC).isoformat(timespec="seconds")
        ROOTPATH = os.path.dirname(__file__)
        MODELSPATH = os.path.join(ROOTPATH, "models")
        self.local_status_path = os.path.join(MODELSPATH, self.status_blob)

        # gcp_related
        configs = Configs(None)  # resources we need do not depend on env
        self.gcp_project = configs.cloud["gcs"]["project"]
        self.gcp_bucket = configs.cloud["gcs"]["mlflow_bucket"]
        self.base_prefix = "runs"

    # Internals
    def _key_for_target_alias(self, target_alias: str) -> str:
        """Map target alias -> key in PromotionStatus ('challenger'|'champion')."""
        alias_norm = target_alias.lower()
        if "challenger" in alias_norm:
            return "challenger"
        if "champion" in alias_norm:
            return "champion"
        raise ValueError(
            f"Unsupported target alias '{target_alias}'. Expected Challenger or Champion."
        )

    def _bundle_path_for_runid(self, run_id: str) -> str:
        return f"{self.base_prefix}/{run_id}"

    def _bundle_uri_for_runid(self, run_id: str) -> str:
        return f"gs://{self.gcp_bucket}/{self._bundle_path_for_runid(run_id)}"

    def _read_metadata_by_runid(self, run_id: str) -> dict:
        # gs://bucket/runs/<run_id>/mlflow_models/metadata.json
        return load_json_from_gcs(
            self.gcp_project,
            self.gcp_bucket,
            f"{self._bundle_path_for_runid(run_id)}/{self.model_artifact_folder}/metadata.json",
        )

    def _load_or_init_status(self) -> PromotionStatus:
        # Prefer GCS
        d = load_json_from_gcs(self.gcp_project, self.gcp_bucket, self.status_blob)
        if d:
            self.logger.info(
                f"Loading status from GCS bucket: {self.gcp_bucket}/{self.status_blob}"
            )
            return self._status_from_dict(d)
        # Else check if it exists locally
        if os.path.exists(self.local_status_path):
            self.logger.info(
                f"Loading status from local filesystem: {self.local_status_path}"
            )
            with open(self.local_status_path, encoding="utf-8") as f:
                return self._status_from_dict(json.load(f))
        # Create new
        self.logger.info("Initializing new status")
        return PromotionStatus(
            last_updated=self.now,
            challenger=None,
            champion=None,
            archived={"challenger": [], "champion": []},
        )

    def _save_status(self, status: PromotionStatus) -> None:
        status.last_updated = self.now
        # local
        with open(self.local_status_path, "w", encoding="utf-8") as f:
            json.dump(status.to_dict(), f, indent=2, sort_keys=True)
        # gcs
        upload_file_to_folder(
            self.gcp_project, self.gcp_bucket, folder="", file=self.local_status_path
        )

    def _status_from_dict(self, d: dict[str, Any]) -> PromotionStatus:
        def ptr_to_mref(ptr: dict[str, Any] | None, alias: str) -> ModelRef | None:
            if not ptr:
                return None
            return ModelRef(
                run_id=ptr["run_id"],
                model_version=ptr.get("model_version"),
                bundle_uri=ptr["bundle_uri"],
                alias=alias,
                last_updated=ptr.get("last_updated"),
            )

        return PromotionStatus(
            last_updated=d.get("last_updated", self.now),
            challenger=ptr_to_mref(d.get("challenger"), "Challenger"),
            champion=ptr_to_mref(d.get("champion"), "Champion"),
            archived=d.get("archived", {"challenger": [], "champion": []}),
        )

    def _ref_from_alias(self, alias: str) -> ModelRef:
        mv = self.client.get_model_version_by_alias(self.registry_name, alias)
        meta = self._read_metadata_by_runid(mv.run_id) or {}
        return ModelRef(
            run_id=mv.run_id,
            model_version=mv.version,
            bundle_uri=self._bundle_uri_for_runid(mv.run_id),
            alias=alias,
            metrics=meta.get("metrics"),
            last_updated=self.now,
        )

    def _archive_incumbent(self, incumbent: ModelRef, *, archived_from: str) -> None:
        """Mark the displaced incumbent as archived in the registry, with immutable breadcrumbs."""
        ts = self.now.replace(":", "").replace("-", "")
        # Transition stage to Archived (shows up clearly in UIs)
        try:
            self.client.transition_model_version_stage(
                name=self.registry_name,
                version=incumbent.model_version,
                stage="Archived",
            )
        except Exception as e:
            self.logger.warning(
                "Stage transition to Archived failed (non-fatal): %s", e
            )

        # Tag with metadata for traceability
        try:
            self.client.set_model_version_tag(
                self.registry_name, incumbent.model_version, "archived_at", self.now
            )
            self.client.set_model_version_tag(
                self.registry_name,
                incumbent.model_version,
                "archived_from",
                archived_from,
            )
        except Exception as e:
            self.logger.warning("Tagging archived version failed (non-fatal): %s", e)

        # immutable alias to use for rollbacks / audits
        try:
            archived_alias = f"archived-{archived_from.lower()}-{ts}"
            self.client.set_registered_model_alias(
                self.registry_name,
                alias=archived_alias,
                version=incumbent.model_version,
            )
        except Exception as e:
            self.logger.warning("Setting archived alias failed (non-fatal): %s", e)

    @staticmethod
    def _is_better(
        insurgent: float,
        incumbent: float,
        *,
        higher_is_better: bool,
        on_equal_promote: bool,
    ) -> bool:
        if insurgent == incumbent:
            return on_equal_promote
        return (insurgent > incumbent) if higher_is_better else (insurgent < incumbent)

    def _is_inusurgent_better_than_incumbent(
        self,
        *,
        source_alias: str,
        target_alias: str,
        insurgent: ModelRef,
        incumbent: ModelRef,
        metric_key: str = "overall_test_rmse",
        higher_is_better: bool = False,  # default "minimize"
        on_equal_promote: bool = False,  # tie-breaker (False = skip on equal)
    ) -> bool:
        # Get the incumbent score
        inc_metrics = incumbent.metrics or {}
        if metric_key not in inc_metrics:
            raise ValueError(
                f"metric_key '{metric_key}' missing in incumbent ({target_alias}) metrics"
            )
        inc_score = inc_metrics[metric_key]

        # Get the insurgent score
        ins_metrics = insurgent.metrics or {}
        if metric_key not in ins_metrics:
            raise ValueError(
                f"metric_key '{metric_key}' missing in insurgent ({source_alias}) metrics"
            )
        ins_score = ins_metrics[metric_key]

        # compare the scores, if insurgent is not better, return
        is_better = self._is_better(
            ins_score,
            inc_score,
            higher_is_better=higher_is_better,
            on_equal_promote=on_equal_promote,
        )
        if is_better:
            result_str = "is better"
        else:
            result_str = "is NOT better"
        self.logger.info(
            "%s → %s: insurgent(%s)=%.6g %s than incumbent(%s)=%.6g [higher_is_better=%s, on_equal_promote=%s]",
            source_alias,
            target_alias,
            metric_key,
            ins_score,
            result_str,
            metric_key,
            inc_score,
            higher_is_better,
            on_equal_promote,
        )
        return is_better

    def _promote(
        self,
        *,
        source_alias: str,
        target_alias: str,
        metric_key: str = "overall_test_rmse",
        higher_is_better: bool = False,  # default "minimize"
        on_equal_promote: bool = False,  # tie-breaker (False = skip on equal)
        require_better: bool = True,
    ) -> tuple[bool, ModelRef]:
        """
        Compare insurgent (source_alias) vs incumbent (target_alias) and promote if better.

        Returns:
          (promoted: bool, target_ref_after: ModelRef)
        """
        # Resolve insurgent(e.g., incoming Candidate or Challenger)
        insurgent = self._ref_from_alias(source_alias)

        # Resolve incumbent (e.g., current Challenger or Champion), if exists
        try:
            incumbent = self._ref_from_alias(target_alias)
        except Exception:
            incumbent = None

        # If nothing changes, exit early
        if (
            incumbent
            and incumbent.run_id == insurgent.run_id
            and incumbent.model_version == insurgent.model_version
        ):
            self.logger.info(
                "No-op: %s already points to run_id=%s ver=%s",
                target_alias,
                incumbent.run_id,
                incumbent.model_version,
            )
            return

        # if we want to promote only if the insurgent is better than an existing incumbent:
        if require_better and incumbent:
            insurgent_better = self._is_inusurgent_better_than_incumbent(
                source_alias=source_alias,
                target_alias=target_alias,
                insurgent=insurgent,
                incumbent=incumbent,
                metric_key=metric_key,
                higher_is_better=higher_is_better,
                on_equal_promote=on_equal_promote,
            )
            if not insurgent_better:
                return

        # Proceed with promotion if require_better, no incumbent available or insurgent is better than incumbent

        # Perform promotion: set alias in MLflow
        self.client.set_registered_model_alias(
            self.registry_name, alias=target_alias, version=insurgent.model_version
        )

        # Update promotion_status.json
        st = self._load_or_init_status()
        key = self._key_for_target_alias(target_alias)

        # Append an archive entry if there was an incumbent
        if incumbent is not None:
            st.archived.setdefault(key, []).append({
                "run_id": incumbent.run_id,
                "model_version": incumbent.model_version,
                "when": self.now,
            })

        # Update the pointer for challenger/champion
        setattr(st, key, insurgent)  # challenger or champion
        st.last_updated = self.now

        # Persist
        self._save_status(st)

        self.logger.info(
            "Promoted %s (run_id=%s, ver=%s) to %s",
            source_alias,
            insurgent.run_id,
            insurgent.model_version,
            target_alias,
        )
        return

    # Public helpers
    def current(self) -> tuple[ModelRef | None, ModelRef | None]:
        st = self._load_or_init_status()
        return st.challenger, st.champion

    def status_dict(self) -> dict[str, Any]:
        return self._load_or_init_status().to_dict()

    def bundle_path_for_runid(self, run_id: str) -> str:
        return f"{self.base_prefix}/{run_id}"

    def promote_candidate_to_challenger(
        self,
        *,
        metric_key: str | None = None,
        higher_is_better: bool = False,
        on_equal_promote: bool = False,
        require_better: bool = True,
    ) -> tuple[bool, ModelRef]:
        """Candidate → Challenger with optional metric gate against current Challenger."""
        self._promote(
            source_alias=self.candidate_alias,
            target_alias=self.challenger_alias,
            metric_key=metric_key,
            higher_is_better=higher_is_better,
            on_equal_promote=on_equal_promote,
            require_better=require_better,
        )

    def promote_challenger_to_champion(
        self,
        *,
        metric_key: str | None = None,
        higher_is_better: bool = False,
        on_equal_promote: bool = False,
        require_better: bool = True,
    ) -> tuple[bool, ModelRef]:
        """Challenger → Champion with optional metric gate against current Champion."""
        self._promote(
            source_alias=self.challenger_alias,
            target_alias=self.champion_alias,
            metric_key=metric_key,
            higher_is_better=higher_is_better,
            on_equal_promote=on_equal_promote,
            require_better=require_better,
        )


def main(
    promote_candidate: bool, promote_challenger: bool, require_better: bool
) -> None:
    promoter = ModelPromoter()

    print("Current promotion status:")
    print(json.dumps(promoter.status_dict(), indent=2))

    # ExpWinner -> Challenger
    if promote_candidate:
        print(f"promoting candidate -> challenger [require_better={require_better}]")
        promoter.promote_candidate_to_challenger(
            metric_key="overall_test_rmse",
            higher_is_better=False,
            on_equal_promote=False,
            require_better=require_better,  # False, if we want to promote anyway
        )

    # Challenger -> Champion
    if promote_challenger:
        print(f"promoting challenger -> champion  [require_better={require_better}]")
        promoter.promote_challenger_to_champion(
            metric_key="overall_test_rmse",
            higher_is_better=False,
            on_equal_promote=False,
            require_better=require_better,  # False, if we want to promote anyway
        )

    if promote_candidate or promote_challenger:
        print("Updated promotion status:")
        print(json.dumps(promoter.status_dict(), indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Promote models from candidate to challenger and from challenger to champion. Example use: python main_promotion.py --promote_candidate"
    )
    parser.add_argument("--promote_candidate", action="store_true")
    parser.add_argument(
        "--no-promote_candidate", dest="promote_candidate", action="store_false"
    )
    parser.add_argument("--promote_challenger", action="store_true")
    parser.add_argument(
        "--no-promote_challenger", dest="promote_challenger", action="store_false"
    )
    parser.add_argument(
        "--no-require_better", dest="require_better", action="store_false"
    )
    parser.add_argument("--require_better", action="store_true")
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    main(args.promote_candidate, args.promote_challenger, args.require_better)
