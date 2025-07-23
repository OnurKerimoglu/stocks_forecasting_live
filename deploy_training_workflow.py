from datetime import datetime, timedelta

from prefect import Flow
from prefect.runner.storage import GitRepository
from prefect.schedules import Interval


def main(mode: str) -> None:
    if mode == "prod":
        branch = "prod"
        schedule = Interval(
            timedelta(weeks=1),
            anchor_date=datetime(2025, 7, 22, 0, 0),
            timezone="Germany/Berlin",
        )
    elif mode == "dev":
        branch = "dev"
        schedule = None

    source = GitRepository(
        url="https://github.com/OnurKerimoglu/stocks_forecasting_live.git",
        branch=branch,
    )

    Flow.from_source(
        source=source, entrypoint="main_training.py:stocks_forecasting_training_flow"
    ).deploy(
        name=f"stocks_forecasting_train_flow_{mode}",
        schedule=schedule,
        work_pool_name="stocks_forecasting_live_local",
    )


if __name__ == "__main__":
    main(mode="dev")
    # main(mode="prod")


# stocks_forecasting_training_flow.serve(
#   name="stocks_forecasting_train_workflow",
#   schedule=Interval(
#     timedelta(weeks=1),
#     anchor_date=datetime(2025, 7, 22, 0, 0),
#     timezone="Germany/Berlin"
#   )
# )

# stocks_forecasting_training_pipeline.deploy(
#     name="stocks_forecasting_train",
#     work_pool_name="stocks_forecasting_live_local",
#     image="my-image",
#     push=False,
#     # cron="* * * * *",
# )
