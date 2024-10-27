"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.8
"""
"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.7
"""

import logging
import typing as t
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import mlflow
import mlflow.sklearn


logger = logging.getLogger(__name__)

# def split_data(df: pd.DataFrame, parameter: t.Dict) -> t.Tuple:
#     """
#     Splits the input dataframe into training and testing datasets.
    
#     Args:
#         df: The input dataframe containing the feature columns and target column 'y'.
#         parameter: A dictionary with configuration parameters, e.g., "test_size".
    
#     Returns:
#         A tuple containing:
#             - X_train: Training feature data.
#             - X_test: Testing feature data.
#             - y_train: Training target data.
#             - y_test: Testing target data.
#     """
#     #X = df[parameter["features"]]
#     X = df.drop(columns=["y"])
#     y = df["y"]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=parameter["test_size"]
#     )
#     return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, model_path: str) -> LogisticRegression:
    """
    Trains a logistic regression model on the provided training data.
    
    Args:
        X_train: Training features.
        y_train: Training target.
    
    Returns:
        The trained logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # if mlflow.active_run():
    #     mlflow.end_run()

    # # Log the model with MLflow
    # with mlflow.start_run():
    #     mlflow.sklearn.log_model(model, "logistic_regression_model")
    #     mlflow.log_params({"model_type": "Logistic Regression"})
    #     mlflow.log_metric("accuracy", model.score(X_train, y_train))

    # Save the trained model
    joblib.dump(model, model_path)  # or use pickle to save
    logger.info(f"Model saved to {model_path}")

    return model

def evaluate_model(regressor: LogisticRegression, X_test: pd.Series, y_test: pd.Series):
    """
    Evaluates the trained model on the test dataset and logs the accuracy score.
    
    Args:
        regressor: The trained logistic regression model.
        X_test: Testing features.
        y_test: Testing target.
    
    Logs:
        The accuracy of the model on the test data.
    """
    y_pred = regressor.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    logger.info("Model has a accuracy of %.3f on test data.", score)
    return y_pred

def plot_confusion_matrix(y_test, y_pred, output_path: str):
    """
    Generates a confusion matrix plot and saves it to the specified path.
    
    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        output_path: Path to save the confusion matrix plot.
    
    Returns:
        None
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # # Plot confusion matrix using seaborn for better visuals
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    
    # # Save the plot
    # plt.savefig(output_path)
    # plt.close()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    return fig
