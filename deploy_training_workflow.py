import argparse
import os
from datetime import datetime, timedelta

from prefect import Flow
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval


def main(env: str) -> None:
    deployment_name = f"stocks_forecasting_train_flow_{env}"
    if env == "prod":
        print(
            f"deploying the train pipeline {deployment_name} from git repository and with schedule"
        )
        schedule = Interval(
            timedelta(weeks=1),
            anchor_date=datetime(2025, 7, 26, 0, 0),
            timezone="Germany/Berlin",
        )
        source = GitRepository(
            url="https://github.com/OnurKerimoglu/stocks_forecasting_live.git",
            branch="prod",
        )
    elif env in ["test", "dev"]:
        print(
            f"deploying the train pipeline {deployment_name} from local filesystem and without schedule"
        )
        schedule = None
        # source is the local file system
        source = os.path.dirname(__file__)

    Flow.from_source(
        source=source, entrypoint="main_training.py:stocks_forecasting_training_flow"
    ).deploy(
        name=deployment_name,
        schedule=schedule,
        work_pool_name="stocks_forecasting_live_local",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the train workflow to prefect")
    parser.add_argument("--env", type=str, required=True, help="dev or prod")
    args = parser.parse_args()
    main(env=args.env)
