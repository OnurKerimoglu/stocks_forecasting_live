import os
from datetime import datetime

import kaggle
import pandas as pd
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def load_raw_data(datapath: str, user: str, datasetname: str) -> pd.DataFrame:
    """
    Loads raw data from a specified dataset using the Kaggle API.

    For the Kaggle API to work, you need to have a Kaggle account and
    a Kaggle API token (see: https://www.kaggle.com/docs/api).
    If the dataset file does not exist in the specified path, it downloads
    and unzips the dataset from Kaggle. Otherwise, it reads the existing
    dataset file. The raw data is returned as a pandas DataFrame.

    Args:
    datapath : str
        The directory path where the dataset is stored or will be downloaded.
    user : str
        The Kaggle username associated with the dataset.
    datasetname : str
        The name of the dataset on Kaggle.

    Returns:
    pd.DataFrame
        A DataFrame containing the raw data from the dataset.
    """

    print(f"rootpath: {datapath}")

    os.makedirs(datapath, exist_ok=True)
    if not os.path.exists(os.path.join(datapath, "World-Stock-Prices-Dataset.csv")):
        kaggle.api.dataset_download_files(
            dataset=f"{user}/{datasetname}", path=datapath, unzip=True
        )
    else:
        print(f"Raw data already found in location {datapath}")

    raw_fpath = os.listdir(datapath)[0]
    raw_fpath_full = os.path.join(datapath, raw_fpath)

    print(f"reading raw data from: {raw_fpath_full}")
    df_raw = pd.read_csv(raw_fpath_full)
    return df_raw


def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
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
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], utc=True).dt.tz_convert(None)
    df_clean.drop_duplicates(subset=["Date", "Ticker"], keep="first", inplace=True)
    df_clean = df_clean[["Date", "Close", "Ticker"]]
    df_clean["returns"] = df_clean.groupby("Ticker")["Close"].pct_change()
    lower, upper = (
        df_clean["returns"].quantile(0.01),
        df_clean["returns"].quantile(0.99),
    )
    print(
        f"The returns are winsorized with upper and lower caps of respectively {upper} and {lower}"
    )
    df_clean["returns"] = df_clean["returns"].clip(lower, upper)
    df_clean.dropna(inplace=True)
    return df_clean


def sample_tickers_dates(
    df_clean: pd.DataFrame,
    tickers: list | None = None,
    startdate: datetime | None = None,
    clean_sample_fpath_full: str | None = None,
) -> pd.DataFrame:
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
    clean_sample_fpath_full : str or None, optional
        The path to write the sampled DataFrame to a CSV file. If None, the sampled
        DataFrame is not written to file. Defaults to None.

    Returns:
    pd.DataFrame
        A new DataFrame with the sampled data, sorted by 'Ticker' and 'Date'.
    """
    if tickers is None:
        df_clean_sample = df_clean.copy()
    else:
        print(f"Sampling tickers: {tickers}")
        df_clean_sample = df_clean[(df_clean["Ticker"].isin(tickers))].copy()
    if startdate is not None:
        print(f"Sampling from start date: {startdate}")
        df_clean_sample = df_clean_sample[df_clean_sample["Date"] >= startdate].copy()
    # df_clean_sample.Date = pd.to_datetime(df_clean_sample['Date'])
    # print(f'sample shape: {df_clean_sample.shape}')
    # df_clean_sample.sort_values('Date', ascending=True).head()
    if clean_sample_fpath_full is not None:
        df_clean_sample.to_csv(clean_sample_fpath_full, index=False)
        print(f"Wrote clean sample to: {clean_sample_fpath_full}")
    df_clean_sample.sort_values(["Ticker", "Date"], inplace=True)
    return df_clean_sample


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
    df_in: pd.DataFrame, lags: int = 3
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

    Returns:
    df_out : pd.DataFrame
        New dataframe with additional feature columns.
    features2scale : List[str]
        List of feature names that should be scaled (e.g. by StandardScaler), depending on the model.
    """
    feats = []
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

        # Trend + seasonality
        df["Period"] = df["Date"].dt.to_period("D")
        df = df.set_index("Period")
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

        # Select features
        feature_cols = [*["Ticker", "Date", "returns"], *lag_feat_names, *ma_feat_names, *index_feat_names]
        df_feat = df[feature_cols]

        # Merge and reset index
        df_feat = df_feat.reset_index().drop(columns=["Period"])
        merged = pd.concat(
            [df_feat.reset_index(drop=True), tf.reset_index(drop=True)], axis=1
        )

        # Drop rows with any NaNs
        merged = merged.dropna().reset_index(drop=True)

        feats.append(merged)

    # Concatenate all ticker dataframes
    df_out = pd.concat(feats, ignore_index=True)
    # print(f'Built features: {df_out.columns}')

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
            print(f"For ticker: {ticker}, no data left after dropping NaNs.")
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
            print(f"X shape: {X_all.shape}, y_multi shape: {y_multi_all.shape}")
        X_all.set_index(["Ticker", "Date"], inplace=True)
        y_multi_all.set_index(["Ticker", "Date"], inplace=True)
        return X_all, y_multi_all
