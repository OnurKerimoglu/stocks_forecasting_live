import logging
import os
from datetime import datetime

import pandas as pd

from gcp_functions import read_file_as_df
from load_configs import Configs
from raw_data_kaggle import kaggle_download_dataset
from raw_data_yf import fetch_ticker_data_from_yf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_raw_data(
    datasource: str,
    datapath: str,
    localrun: bool,
    env: str,
    user: str,
    datasetname: str,
) -> tuple[pd.DataFrame, str]:
    """
    Loads raw data from given datasource.
    Args:
    datasource: str
        The source of the raw data. Currently, only 'kaggle' and 'yahoofinance' are supported.
    datapath: str
        The path where the raw data will be stored
    localrun: bool
        Whether the pipeline is running locally or not.
    env: str
        The environment in which the pipeline is running.
    user: str
        The username of the user from which the dataset is fetched.
    datasetname: str
        The name of the dataset to be fetched.
    Returns:
    tuple[pd.DataFrame, str]
        A tuple containing the raw data as a pandas DataFrame and the date when the data was accessed (as a string in ISO format).
    """
    if datasource == "kaggle":
        df_raw, access_date_str = load_raw_data_from_kaggle(
            parentdatapath=datapath,
            user=user,
            datasetname=datasetname,
        )
    elif datasource == "yahoofinance":
        df_raw, access_date_str = load_raw_data_from_yf(
            parentdatapath=datapath, localrun=localrun, env=env
        )
    else:
        raise ValueError(f"Unknown datasource: {datasource}")

    return df_raw, access_date_str


def load_raw_data_from_kaggle(
    parentdatapath: str, user: str, datasetname: str
) -> tuple[pd.DataFrame, str]:
    """
    Loads raw data from a specified dataset using the Kaggle API.

    For the Kaggle API to work, you need to have a Kaggle account and
    a Kaggle API token (see: https://www.kaggle.com/docs/api).
    If the dataset file does not exist in the specified path, it downloads
    and unzips the dataset from Kaggle. Otherwise, it reads the existing
    dataset file. The raw data is returned as a pandas DataFrame.

    Args:
    parentdatapath : str
        The directory path where a "kaggle" folder, and inside which the dataset is expected to be found.
    user : str
        The Kaggle username associated with the dataset.
    datasetname : str
        The name of the dataset on Kaggle.

    Returns:
    pd.DataFrame
        A DataFrame containing the raw data from the dataset.
    access_date_str : str
        The date string of when the raw data was accessed.
    """
    datapath = os.path.join(parentdatapath, "raw", "kaggle")
    logger.info(f"kaggle datapath: {datapath}")
    os.makedirs(datapath, exist_ok=True)

    raw_fpath_full = os.path.join(datapath, "World-Stock-Prices-Dataset.csv")
    if not os.path.exists(raw_fpath_full):
        kaggle_download_dataset(user, datasetname, datapath)
        access_date_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"Raw data was not found in location {datapath}, downloading from kaggle"
        )
    else:
        access_date_timestamp = os.path.getmtime(raw_fpath_full)
        access_date_str = datetime.fromtimestamp(access_date_timestamp).strftime(
            "%Y-%m-%d"
        )
        logger.info(
            f"Raw data already found in location {datapath}, last modified on {access_date_str}"
        )

    logger.info(f"reading raw data from: {raw_fpath_full}")
    df_raw = pd.read_csv(raw_fpath_full)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], utc=True).dt.tz_convert(None)
    return df_raw, access_date_str


def load_raw_data_from_yf(
    parentdatapath: str, localrun: bool, env: str
) -> tuple[pd.DataFrame, str]:
    """
    Loads raw data from a specified dataset using yfinance.

    If the merged dataset (csv) file does not exist in the specified local path, it:
    - reads the ref data to find out each ticker to fetch
    - fetches the data for each ticker from yfinance
    - merges the ticker data in a dataframe and stores as a csv file
    Otherwise it reads the data from the local csv file
    dataset file. The raw data is returned as a pandas DataFrame.

    Args:
    parentdatapath : str
        The directory path where a "yf" folder, and inside which the dataset is expected to be found.
    localrun: bool
        If True, the ref data is found in local filesystem, otherwise in cloud
    env: str
        Environment (if localrun is false, determines the bucket where the ref data is found)

    Returns:
    pd.DataFrame
        A DataFrame containing the raw data from the dataset.
    access_date_str : str
        The date string of when the raw data was accessed.
    """
    datapath = os.path.join(parentdatapath, "raw", "yf")
    logger.info(f"yf datapath: {datapath}")
    os.makedirs(datapath, exist_ok=True)

    raw_fpath_full = os.path.join(datapath, "Stock-Prices-Ref-Dataset_YF.csv")
    if not os.path.exists(raw_fpath_full):
        access_date_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"Raw data was not found in location {datapath}, downloading from yf"
        )
        # First find out which tickers are needed, by reading the reference data
        configs = None if localrun else Configs(env)
        # load reference data
        ref_data = load_data(
            localrun,
            prefix="ref_data_model",
            fname="data.parquet",
            project=configs.cloud["gcs"]["project"] if configs else None,
            bucket=configs.cloud["gcs"]["data_monitoring_bucket"] if configs else None,
            localrootdir=parentdatapath,
        )
        # For each ticker, fetch data from yf, merge and store as a csv file
        read_merge_store_yf_data(ref_data, raw_fpath_full)
    else:
        access_date_timestamp = os.path.getmtime(raw_fpath_full)
        access_date_str = datetime.fromtimestamp(access_date_timestamp).strftime(
            "%Y-%m-%d"
        )
        logger.info(
            f"Raw data already found in location {datapath}, last modified on {access_date_str}"
        )
    logger.info(f"reading raw data from: {raw_fpath_full}")
    df_raw = pd.read_csv(raw_fpath_full)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"], utc=True).dt.tz_convert(None)
    return df_raw, access_date_str


def read_merge_store_yf_data(
    ref_data: pd.DataFrame, raw_fpath_full: str, fail_tolerance: int = 10
) -> None:
    tickers = ref_data["Ticker"].unique()
    df_raw = None
    fail_count = 0
    for i, ticker in enumerate(tickers):
        try:
            df_ticker = fetch_ticker_data_from_yf(
                ticker,
                fetchperiodinweeks=52 * 4 + 1,  # i.e., 4 years plus 1 week
            )
        except ValueError as err:
            fail_count += 1
            logger.error(f"Error fetching data for ticker # {i} ({ticker}): {err}")
        if df_raw is None:
            df_raw = df_ticker
        else:
            df_raw = pd.concat([df_raw, df_ticker])
    fail_perc = fail_count / len(tickers) * 100
    succ_perc = 100 - fail_perc
    if fail_perc > fail_tolerance:
        raise ValueError(
            f"Failed to fetch data for {fail_perc:.2f}% (>{fail_tolerance}%) of the tickers found in reference data"
        )
    else:
        logger.info(
            f"Successfully fetched data for {succ_perc:.2f}% of the tickers found in reference data"
        )
    df_raw.to_csv(raw_fpath_full, index=False)
    return df_raw


def remove_raw_data(rawdatapath: str, datasource: str) -> None:
    """
    Removes the raw data from the specified path if it exists.

    Args:
    datapath : str
        The directory path where the raw data is stored.
    datasource : str
        The source of the raw data (only "kaggle" is supported for now)

    Returns:
    None
    """
    if datasource == "kaggle":
        raw_data_fpath = os.path.join(
            rawdatapath, "kaggle", "World-Stock-Prices-Dataset.csv"
        )
        if os.path.exists(raw_data_fpath):
            logger.info(f"Removing the raw data @ {raw_data_fpath}")
            os.remove(raw_data_fpath)
        else:
            logger.info(f"Raw data not found @ {raw_data_fpath}")
    else:
        raise ValueError(f"Unknown datasource: {datasource}")


def sample_tickers_dates(
    df_clean: pd.DataFrame,
    tickers: list | None = None,
    startdate: datetime | None = None,
    datasource: str | None = None,
    clean_sample_fdir: str | None = None,
    access_date_str: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Samples a subset of tickers and/or dates from a cleaned DataFrame.

    This function takes a cleaned DataFrame of stock price data and samples a subset
    of tickers and/or dates, returning a new DataFrame with the sampled data.

    Args:
    df_clean : pd.DataFrame
        The cleaned DataFrame of stock price data, which should have columns
        'Date', 'Close', and 'Ticker'.
    tickers : list or None, optional
        A list of tickers to sample from the data. If None, all tickers are kept.
        Defaults to None.
    startdate : datetime or None, optional
        The start date to sample from the data. If None, all dates are kept.
        Defaults to None.
    datasource: src, optional
        The source of the data. E.g., 'Kaggle', 'yahoofinance
    clean_sample_fdir : str or None, optional
        Path to the local directory in which the sampled DataFrame will be stored as CSV.
        If None (default), the DataFrame is not written to file.
    access_date_str : str or None, optional
        The date of the data access.
        Defaults to None

    Returns:
    pd.DataFrame
        A new DataFrame with the sampled data, sorted by 'Ticker' and 'Date'.
    str
        The file name of the sampled DataFrame in the local directory.
    """
    if tickers is None:
        df_clean_sample = df_clean.copy()
        ticker_suffix = "WSPall"
    else:
        logger.info(f"Sampling tickers: {tickers}")
        df_clean_sample = df_clean[(df_clean["Ticker"].isin(tickers))].copy()
        ticker_suffix = "Tickers_" + "-".join(tickers)
    if startdate is not None:
        logger.info(f"Sampling from start date: {startdate}")
        df_clean_sample = df_clean_sample[df_clean_sample["Date"] >= startdate].copy()
        sample_date_suffix = f"_from_{startdate.strftime('%Y-%m-%d')}"
    else:
        sample_date_suffix = ""

    # df_clean_sample.Date = pd.to_datetime(df_clean_sample['Date'])
    # logger.info(f'sample shape: {df_clean_sample.shape}')
    # df_clean_sample.sort_values('Date', ascending=True).head()
    # construct an id for archiving data
    if clean_sample_fdir is not None:
        access_date_suffix = f"Access_{access_date_str}"
        fname_root = (
            f"{datasource}_{access_date_suffix}_{ticker_suffix}{sample_date_suffix}"
        )
        fpath = store_df_locally(df_clean_sample, fname_root, clean_sample_fdir)
        logger.info(f"Wrote cleaned sample to: {fpath}")
    else:
        fpath = None
    df_clean_sample.sort_values(["Ticker", "Date"], inplace=True)

    return df_clean_sample, fpath


def store_df_locally(
    df: pd.DataFrame, fname_root: str, local_fdir: str, format: str = "parquet"
) -> str:
    os.makedirs(local_fdir, exist_ok=True)
    if format == "parquet":
        fpath = os.path.join(local_fdir, f"{fname_root}.parquet")
        df.to_parquet(fpath)
    elif format == "csv":
        fpath = os.path.join(local_fdir, f"{fname_root}.csv")
        df.to_csv(fpath)
    return fpath


def load_data(
    localrun: bool,
    prefix: str,
    fname: str,
    project: str | None,
    bucket: str | None,
    localrootdir: str | None = None,
) -> pd.DataFrame:
    if not localrun:
        # Load the df from GCS
        logger.info(f"Loading data from GCS bucket: {bucket}")
        df = read_file_as_df(project, bucket, f"{prefix}/{fname}")
    else:
        # Read from the local filesystem
        fpath = os.path.join(localrootdir, prefix, fname)
        logger.info(f"Loading data from filesystem: {fpath}")
        if not os.path.exists(fpath):
            raise Exception(
                f"no config provided for GCS and local path {fpath} does not exist"
            )
        else:
            df = pd.read_parquet(fpath)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )
    # load the raw data
    ROOTPATH = os.path.dirname(os.path.dirname(__file__))
    DATAPATH = os.path.join(ROOTPATH, "data")
    df_raw, access_date_str = load_raw_data(
        # datasource="kaggle",
        datasource="yahoofinance",
        datapath=DATAPATH,
        localrun=False,
        env="test",
        user="nelgiriyewithana",
        datasetname="world-stock-prices-daily-updating",
    )
