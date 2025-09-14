import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        raise ValueError(f"Failed to fetch data for {ticker}") from e

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
        logger.error(f"No valid rows in fetched data for {ticker}")
        raise ValueError(f"No valid rows in fetched data for {ticker}")
    return df
