import logging

import kaggle

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def kaggle_download_dataset(user: str, datasetname: str, datapath: str) -> None:
    kaggle.api.dataset_download_files(
        dataset=f"{user}/{datasetname}", path=datapath, unzip=True
    )
