import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import RegressorChain
from sklearn.utils.validation import check_is_fitted

from data import build_features, create_X_y_multistep
from models import create_xgbregressor_chain


class PpModelPl(BaseEstimator, RegressorMixin):
    """
    Preprocessor - Model- Pipeline that operates with raw data,
    providing sklearn-style fit/predict interface

    Parameters
    ----------
    target : str
        Name of target column used by create_X_y_multistep (e.g., 'returns').
    steps : int
        Forecast horizon used in create_X_y_multistep.
    date_col : str
        E.g., 'Date'
    price_col : str
        Column to compute returns from (e.g., 'Close').
    ticker_col : str
        Column that holds ticker symbols (e.g., 'Ticker').
    winsorize : bool
        Whether to winsorize target column in cleaning.
    q_low, q_high : float
        Winsor quantiles computed on training rows.
    lags : int
        Number of lags of the target variable to be included in features
    CldrFeats : bool
        Whether calendar features (seasonality) should be included in features
    ModReg: bool
        Whether a regularization should be applied when instantiating the model
    """

    def __init__(
        self,
        target: str = "returns",
        steps: int = 1,
        date_col: str = "Date",
        price_col: str = "Close",
        ticker_col: str = "Ticker",
        winsorize: bool = True,
        q_low: float = 0.01,
        q_high: float = 0.99,
        lags: int = 3,
        CldrFeats: bool = False,
        ModReg: bool = True,
    ) -> None:
        self.target = target
        self.steps = steps
        self.date_col = date_col
        self.price_col = price_col
        self.ticker_col = ticker_col
        self.winsorize = winsorize
        self.q_low = q_low
        self.q_high = q_high
        self.lags = lags
        self.CldrFeats = CldrFeats
        self.ModReg = ModReg

        # learned in fit
        self._winsor_lo_: float | None = None
        self._winsor_hi_: float | None = None
        self._y_cols_: list | None = None
        self.feature_cols_: list | None = None
        self.estimator_: RegressorChain | None = None

        # handy debug info
        self.clean_stats_: dict = {}

    # helpers
    def _compute_returns(self, df: pd.DataFrame) -> pd.Series:
        returns = df.groupby(self.ticker_col)[self.price_col].pct_change()
        return returns

    def _dedupe_selectcol(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df.drop_duplicates(
            subset=[self.date_col, self.ticker_col], keep="first", inplace=True
        )
        df = df[[self.date_col, self.price_col, self.ticker_col]]
        self.clean_stats_["deduped_rows"] = before - len(df)
        return df

    def _winsor_fit(self, series: pd.Series) -> None:
        self._winsor_lo_ = float(series.quantile(self.q_low))
        self._winsor_hi_ = float(series.quantile(self.q_high))

    def _winsor_apply(self, series: pd.Series) -> pd.Series:
        if not self.winsorize:
            return series
        return series.clip(self._winsor_lo_, self._winsor_hi_)

    def _clean_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        dfc = self._dedupe_selectcol(dfc)

        # compute returns
        dfc[self.target] = self._compute_returns(dfc)

        # train-only winsor thresholds
        if self.winsorize:
            dfc[self.target] = self._winsor_apply(dfc[self.target])

        # drop NA rows (returns introduces NA at first diff)
        before = len(dfc)
        dfc = dfc.dropna()
        self.clean_stats_["dropna_rows"] = before - len(dfc)

        return dfc

    def _clean_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        dfc = self._dedupe_selectcol(dfc)
        dfc[self.target] = self._compute_returns(dfc)
        if self.winsorize:
            dfc[self.target] = self._winsor_apply(dfc[self.target])
        dfc = dfc.dropna()
        return dfc

    def _make_X_y(
        self, df: pd.DataFrame, *, train: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        If train=True: learn winsor caps on this df (used ONLY inside fit()).
        Else: apply previously learned caps.
        """
        if train:
            df_clean = self._clean_fit(df)
        else:
            df_clean = self._clean_transform(df)

        df_feats, _ = build_features(
            df_clean,
            lags=self.lags,
            CldrFeats=self.CldrFeats,
        )
        X, y = create_X_y_multistep(df_feats, steps=self.steps, target=self.target)
        # At predict time, align to training features
        if not train and hasattr(self, "feature_cols_"):
            X = X.reindex(columns=self.feature_cols_)
        return X, y

    # public hook
    def make_X_y(self, df_raw: pd.DataFrame) -> tuple:
        """Build aligned (X, y) using fitted cleaning/FeatureEngineering/windowing"""
        check_is_fitted(self, ["feature_cols_", "estimator_"])
        X, Y = self._make_X_y(df_raw, train=False)
        return X, Y

    # sklearn API
    def fit(self, df_raw: pd.DataFrame) -> object:
        """
        df_raw: full time series up to 'train end'
        """
        # create features (X) and y
        X, y = self._make_X_y(df_raw, train=True)

        # train model
        self.estimator_ = create_xgbregressor_chain(X, y, self.ModReg)
        self.estimator_.fit(X, y)

        # cache feature/target columns for alignment
        self._feature_cols_ = list(X.columns)
        self._y_cols_ = list(y.columns)
        self.clean_stats_["rows"] = len(X)
        self.clean_stats_["y_cols"] = self._y_cols_
        return self

    # sklearn API
    def predict(self, df_raw: pd.DataFrame) -> object:
        """
        df_raw should contain at least the last window needed to build features and lags.
        """
        # create features (X) and y
        X, _ = self._make_X_y(df_raw, train=False)

        # predict
        y_hat = self.estimator_.predict(X)
        # y_hat = pd.DataFrame(y_hat, columns=self._y_cols_)
        return y_hat
