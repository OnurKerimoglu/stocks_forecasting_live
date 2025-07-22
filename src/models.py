
import pandas as pd
from prefect import task
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor


@task(task_run_name='create_fit_xgbregressor_chain')
def create_fit_xgbregressor_chain(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> RegressorChain:
    """
    Instantiates and trains an XGBoost RegressorChain model on the provided training data.

    This function creates an XGBoost regressor with specified hyperparameters
    and wraps it in a RegressorChain to handle multi-output regression.
    The model is then trained using the provided training data.

    Args:
    X_train : pd.DataFrame or np.ndarray
        The input features for training the model.
    y_train : pd.DataFrame or np.ndarray
        The target values corresponding to the input features.

    Returns:
    RegressorChain
        A fitted RegressorChain model using the XGBoost regressor as the base estimator.
    """

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,  # large upper bound
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1e-2,
        reg_lambda=1.0,
        random_state=42,
    )
    estimator = RegressorChain(estimator=xgb)
    estimator.fit(X_train, y_train)
    return estimator


@task(task_run_name='evaluate_all')
def evaluate_all(
    estimator: RegressorChain,  # add other estimators as needed
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    df: pd.DataFrame,
    tickers: list,
) -> dict:
    """
    Loops through the data to perform evalation of the model on the entire datast and on individual tickers.

    The actual evaluation is performed using the `evaluate` function.

    Args:
    estimator : RegressorChain
        The model to be evaluated
    X_train: pd.DataFrame
        Feature data for training
    y_train: pd.DataFrame
        Target data for training
    X_test: pd.DataFrame
        Feature data for testing
    y_train: pd.DataFrame
        Target data for testing
    df : pd.DataFrame
        The original data frame, required for visualization
    tickers : list
        The list of tickers to evaluate the model on

    Returns:
    dict
        A dictionary with the evaluation results for each ticker and overall,
        where each key has a dictionary with the keys 'train_rmse' and 'test_rmse'
        containing the respective root mean squared error.
    """
    y_train_hat = estimator.predict(X_train)
    y_test_hat = estimator.predict(X_test)
    # index for visualisation
    df_indexed = df.set_index(["Ticker", "Date"])
    y_train_hat_df = pd.DataFrame(
        y_train_hat, index=y_train.index, columns=y_train.columns
    )
    y_test_hat_df = pd.DataFrame(y_test_hat, index=y_test.index, columns=y_test.columns)
    scores = {"overall": evaluate(y_train, y_train_hat, y_test, y_test_hat)}
    for ticker in tickers:
        scores[ticker] = evaluate(
            y_train.loc[ticker],
            y_train_hat_df.loc[ticker],
            y_test.loc[ticker],
            y_test_hat_df.loc[ticker],
            df_indexed,
        )
    # train_rmse = scores['overall']['train_rmse']
    # test_rmse = scores['overall']['test_rmse']
    # print((f"Train RMSE: {train_rmse:.5f}\n" f"Test RMSE: {test_rmse:.5f}"))
    return scores


def evaluate(
    y_train: pd.DataFrame,
    y_train_hat: pd.DataFrame,
    y_test: pd.DataFrame,
    y_test_hat: pd.DataFrame,
    df_indexed: pd.DataFrame = None,
) -> dict:
    """
    Evaluate model performance on given train and test data.

    Args:
    y_train, y_train_hat : pd.DataFrame
        Actual and predicted returns for the training data
    y_test, y_test_hat : pd.DataFrame
        Actual and predicted returns for the test data
    df_indexed : pd.DataFrame (optional)
        Indexed (by ticker and date) dataframe for visualisation (not yet implemented)

    Returns:
    scores : dict
        A dictionary containing the train and test root mean squared errors
    """
    scores = {
        "train_rmse": root_mean_squared_error(y_train, y_train_hat),
        "test_rmse": root_mean_squared_error(y_test, y_test_hat),
    }
    if df_indexed is not None:
        pass
        # TODO: implement visualisation
    return scores
