import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

from gcp_functions import read_file_as_df
from load_configs import Configs

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
    import kaggle

    datapath = os.path.join(parentdatapath, "raw", "kaggle")
    logger.info(f"kaggle datapath: {datapath}")
    os.makedirs(datapath, exist_ok=True)

    raw_fpath_full = os.path.join(datapath, "World-Stock-Prices-Dataset.csv")
    if not os.path.exists(raw_fpath_full):
        kaggle.api.dataset_download_files(
            dataset=f"{user}/{datasetname}", path=datapath, unzip=True
        )
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


def fetch_ticker_data_from_yf(
    ticker: str, fetchperiodinweeks: int = 14
) -> pd.DataFrame:
    period_start = datetime.now() - timedelta(weeks=fetchperiodinweeks)
    logger.info(f"Fetching price data for {ticker}")
    try:
        df = yf.download(
            ticker,
            start=period_start,
            end=datetime.now(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        raise e from ValueError(f"Failed to fetch data for {ticker}")

    # Get rid of the redundant Ticker index
    df.columns = df.columns.droplevel("Ticker")
    df.columns.name = None
    # Re-introduce the ticker as a regular column
    df["Ticker"] = ticker
    # Make 'Date' a regular column by resetting the index
    df.reset_index(inplace=True)

    if df.shape[0] > 0:
        logger.info(f"Fetched {df.shape[0]} rows for {ticker}")
    else:
        raise ValueError(f"No valid rows in fetched data for {ticker}")
    return df


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


def clean_raw_data(df_raw: pd.DataFrame, winsorize: bool = True) -> pd.DataFrame:
    """
    Cleans and processes raw stock price data.

    This function takes a DataFrame containing raw stock price data and performs
    a series of cleaning and processing steps, including converting date values,
    removing duplicates, calculating returns, and winsorizing the returns.

    Args:
    df_raw : pd.DataFrame
        The raw stock price data, which should have columns 'Date', 'Close', and 'Ticker'.

    Returns:
    pd.DataFrame
        A cleaned DataFrame with columns 'Date', 'Close', 'Ticker', and 'returns',
        where 'returns' represents the percentage change in 'Close' prices, winsorized
        to remove extreme outliers.
    """
    df_clean = df_raw.copy()
    df_clean.drop_duplicates(subset=["Date", "Ticker"], keep="first", inplace=True)
    df_clean = df_clean[["Date", "Close", "Ticker"]]
    df_clean["returns"] = df_clean.groupby("Ticker")["Close"].pct_change()
    if winsorize:
        lower, upper = (
            df_clean["returns"].quantile(0.01),
            df_clean["returns"].quantile(0.99),
        )
        logger.info(
            f"The returns are winsorized with upper and lower caps of respectively {upper} and {lower}"
        )
        df_clean["returns"] = df_clean["returns"].clip(lower, upper)
    df_clean.dropna(inplace=True)
    return df_clean


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


def split_train_test_panel(
    df: pd.DataFrame, train_ratio: float, date_col: str = "Date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a panel DataFrame into train/test by date, preserving all tickers
    and *not* shuffling.

    Args:
    df : pd.DataFrame
        Your panel data, either indexed by a DatetimeIndex, or containing
        a date column (specified by date_col).
    train_ratio : float
        Fraction of unique dates to use for training (e.g. 0.8).
    date_col : str, optional
        Name of the column containing dates, if df.index is not datetime.

    Returns:
    df_train, df_test : (pd.DataFrame, pd.DataFrame)
        Two DataFrames containing the first train_ratio of dates, and the
        remaining dates, respectively.
    """
    # Extract dates
    dates = pd.to_datetime(df[date_col])

    # Determine split boundary
    unique_dates = pd.Index(dates).unique().sort_values()
    n_train = int(len(unique_dates) * train_ratio)
    if n_train < 1 or n_train >= len(unique_dates):
        raise ValueError("train_ratio produces empty train or test set")
    split_date = unique_dates[n_train - 1]

    # Boolean mask and split
    mask = dates <= split_date
    df_train = df.loc[mask]
    df_test = df.loc[~mask]

    return df_train, df_test


def build_features(
    df_in: pd.DataFrame, lags: int = 3, CldrFeats: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """
    Builds features for a given panel dataframe.

    Features include the specified number of lagged returns,
    Simple Moving Averages (SMA) and Exponential Moving Averages (EMA),
    and Common Indices.

    Args:
    df_in : pd.DataFrame
        Input panel dataframe with columns 'Ticker', 'Date', 'Close', 'returns'.
    lags : int, optional
        Number of lagged features to generate. Defaults to 3.
    CldrFeats: bool, optional
        Whether to include the Calendear Features

    Returns:
    df_out : pd.DataFrame
        New dataframe with additional feature columns.
    features2scale : List[str]
        List of feature names that should be scaled (e.g. by StandardScaler), depending on the model.
    """
    feats = []
    if CldrFeats:
        logger.info("Calendar features will be included")
    else:
        logger.info("Calendar features will NOT be included")
    for _ticker, grp in df_in.groupby("Ticker"):
        df = grp.sort_values("Date").copy()

        features_to_scale = []
        # AR features
        lag_feat_names = []
        for lag in range(1, lags + 1):
            feat_name = f"returns_lag{lag}"
            df[feat_name] = df["returns"].shift(lag)
            lag_feat_names.append(feat_name)
            features_to_scale.append(feat_name)

        # Moving Averages (SMA & EMA)
        ma_feat_names = []
        for w in [10, 50]:
            feat_name = f"SMA_{w}"
            df[feat_name] = df["Close"].rolling(window=w).mean()
            ma_feat_names.append(f"SMA_{w}")
            features_to_scale.append(feat_name)
        for w in [12, 26]:
            feat_name = f"EMA_{w}"
            df[feat_name] = df["Close"].ewm(span=w, adjust=False).mean()
            ma_feat_names.append(feat_name)
            features_to_scale.append(feat_name)

        # Indices
        index_feat_names = []
        # MACD, Signal & Histogram
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        index_feat_names += ["MACD", "MACD_Signal", "MACD_Hist"]
        features_to_scale += ["MACD", "MACD_Signal", "MACD_Hist"]

        # Bollinger Bands & Width
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_STD"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_STD"]
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_STD"]
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        index_feat_names += ["BB_Middle", "BB_STD", "BB_Upper", "BB_Lower", "BB_Width"]
        features_to_scale += [
            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
        ]  # no need to scale BB_STD and BB_Width

        # RSI (14)
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI_14"] = 100 - (100 / (1 + rs))
        index_feat_names += ["RSI_14"]
        # no need to scale this feature

        # Rate of Change (10)
        df["ROC_10"] = df["Close"].pct_change(periods=10)
        index_feat_names += ["ROC_10"]
        features_to_scale += ["ROC_10"]

        # might be needed for aligning with calendar features
        df["Period"] = df["Date"].dt.to_period("D")
        df = df.set_index("Period")

        # Select features
        feature_cols = [
            *["Ticker", "Date", "returns"],
            *lag_feat_names,
            *ma_feat_names,
            *index_feat_names,
        ]
        df_feat = df[feature_cols]

        # Calendar Features
        if CldrFeats:
            dp = DeterministicProcess(
                index=df.index,
                constant=False,
                order=0,  # no trend
                seasonal=False,  # no additional seasonality terms
                additional_terms=[
                    # CalendarFourier(freq='YE', order=1),
                    # CalendarFourier(freq='QE', order=1),
                    CalendarFourier(freq="ME", order=1),
                    CalendarFourier(freq="W", order=1),
                ],
            )
            tf = dp.in_sample()

            # Merge and reset index
            df_feat = df_feat.reset_index().drop(columns=["Period"])
            merged = pd.concat(
                [df_feat.reset_index(drop=True), tf.reset_index(drop=True)], axis=1
            )
        else:
            df_feat = df_feat.reset_index().drop(columns=["Period"])
            merged = df_feat

        # Drop rows with any NaNs
        merged = merged.dropna().reset_index(drop=True)

        feats.append(merged)

    # Concatenate all ticker dataframes
    df_out = pd.concat(feats, ignore_index=True)
    built_features = ", ".join(df_out.columns)
    logger.info(f"Built features: {built_features}")

    return df_out, features_to_scale


def make_multistep_target(y: pd.Series, steps: int) -> pd.DataFrame:
    """
    Generates a multi-step target DataFrame from a single-step target series.

    Args:
    y : pd.Series
        A pandas Series representing the original single-step target values.
    steps : int
        The number of future steps to predict. This determines the number of
        columns in the resulting DataFrame, each representing a future step.

    Returns:
    pd.DataFrame
        A DataFrame with each column corresponding to a future step's target
        values. Rows with NaN values are dropped to ensure data integrity.
    """
    y_multi = pd.concat({f"y_step_{i + 1}": y.shift(-i) for i in range(steps)}, axis=1)
    y_multi.dropna(inplace=True)
    return y_multi


def create_X_y_multistep(
    df_all: pd.DataFrame, steps: int = 5, target: str = "returns", verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates featuress and multi-step targets.

    The function will loop through the tickers and create two new DataFrames:
    First is a df with the same columns as the input features,
    but aligned with the multi-step targets.
    Second is adf with multi-step targets, created by shifting the original
    target series and dropping any resulting NaN values.

    Args:
    df_all : pd.DataFrame
        A pandas DataFrame containing the features and single-step target
        series.
    steps : int
        The number of future steps to predict. This determines the number of
        columns in the resulting DataFrame.
    target : str
        The name of the target column in the input DataFrame.
    verbose : bool
        If True, print shape information about the created DataFrames.

    Returns:
    X_all : pd.DataFrame
        A DataFrame with the same columns as the input features, but with
        each column representing a future step's target values.
    y_multi_all : pd.DataFrame
        A DataFrame with each column corresponding to a future step's target
        values. The index is a MultiIndex of Ticker and Date.
    """
    y_list = []
    X_list = []
    # loop over tickers to create multistep targets
    for ticker, grp in df_all.groupby("Ticker"):
        df = grp.sort_values("Date").copy()
        y = df[target]
        y_multi = make_multistep_target(y, steps=steps).dropna()
        X = df.drop(columns=[target])
        # Shifting has created indexes that don't match. Only keep times for
        # which we have both targets and features.
        y_multi, X = y_multi.align(X, join="inner", axis=0)
        # Add Ticker and Date, which will be used as indices later
        y_multi["Ticker"] = ticker
        y_multi["Date"] = X["Date"]
        # check whether anything left from X and y_multi after droppping Nas
        if y_multi.shape[0] == 0 or X.shape[0] == 0:
            logger.info(f"For ticker: {ticker}, no data left after dropping NaNs.")
        else:
            y_list.append(y_multi)
            X_list.append(X)
    if len(y_list) == 0 or len(X_list) == 0:
        raise ValueError(
            "No data left after processing. Check your input data and parameters."
        )
    else:
        y_multi_all = pd.concat(y_list)
        X_all = pd.concat(X_list)
        if verbose:
            logger.info(f"X shape: {X_all.shape}, y_multi shape: {y_multi_all.shape}")
        X_all.set_index(["Ticker", "Date"], inplace=True)
        y_multi_all.set_index(["Ticker", "Date"], inplace=True)
        return X_all, y_multi_all


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
