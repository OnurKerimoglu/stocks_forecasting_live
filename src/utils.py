import datetime
import logging
import os

import pandas as pd
import tomli
import yfinance as yf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


def get_pipreqs_from_pyproject(path: str) -> list[str]:
    """
    Parses your dependencies from pyproject.toml and returns them as a list
    of pip-style requirement strings.
    """
    with open(path, "rb") as f:
        pyproject = tomli.load(f)

    deps = pyproject.get("project", {}).get("dependencies", [])

    return deps


def fetch_ticker_data_from_yf(ticker: str) -> pd.DataFrame:
    period_start = datetime.datetime.now() - datetime.timedelta(days=100)
    logger.info(f"Fetching price data for {ticker}")
    try:
        df = yf.download(
            ticker,
            start=period_start,
            end=datetime.datetime.now(),
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
        raise ValueError(f"No valid raws in fetched data for {ticker}")
    return df


if __name__ == "__main__":
    rootpath = os.path.dirname(os.path.dirname(__file__))
    pipreqs = get_pipreqs_from_pyproject(os.path.join(rootpath, "pyproject.toml"))
    print(pipreqs)
