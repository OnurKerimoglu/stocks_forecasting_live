import argparse
import os
from datetime import datetime, timedelta

from prefect import Flow
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval


def main(env: str) -> None:
    assert env in ["test", "dev", "prod"]
    deployment_name = f"stocks_forecasting_train_flow_{env}"
    print(f"deploying the train pipeline: {deployment_name}", end="")
    # determine schedule
    if env in ["test", "dev"]:
        print(" without schedule", end="")
        schedule = None
    elif env == "prod":
        print(" with weekly schedule", end="")
        schedule = Interval(
            timedelta(weeks=1),
            anchor_date=datetime(2025, 7, 26, 5, 0),
            timezone="Germany/Berlin",
        )

    # determine source
    if env == "test":
        print(" from the local filesystem")
        rootpath = os.path.dirname(os.path.dirname(__file__))
        source = rootpath  # this is where the main_training.py is expected to be found
    elif env in ["dev", "prod"]:
        print(f" from git repository, {env} branch")
        source = GitRepository(
            url="https://github.com/OnurKerimoglu/stocks_forecasting_live.git",
            branch=env,
        )

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
