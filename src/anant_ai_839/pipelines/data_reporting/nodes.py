"""
This is a boilerplate pipeline 'data_reporting'
generated using Kedro 0.19.8
"""
# src/anant_ai_839/nodes/data_reporting/generate_drift_report.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# src/anant_ai_839/nodes/data_reporting/evaluate_model_performance.py
from sklearn.metrics import classification_report
import pandas as pd

# src/anant_ai_839/nodes/data_reporting/visualize_results.py
import plotly.express as px

import mlflow
import mlflow.sklearn  # For logging sklearn models if needed

import pdfkit
import os

def generate_drift_report(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    """Generate a report on data drift.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        output_path (str): Path to save the report.
    """
    report = Report(metrics=[DataDriftPreset(),])
    #report.calculate(train_data, test_data)
    report.run(current_data=train_data, reference_data=test_data, column_mapping=None)
    
    #saving report as html
    report.save_html(output_path)
    
    # Log the report artifact to MLflow
    mlflow.log_artifact(output_path)
    
def evaluate_model_performance(y_true, y_pred, output_path: str) -> None:
    """Evaluate model performance and save the classification report.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_path (str): Path to save the report.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(output_path)

    # Log the classification report artifact to MLflow
    mlflow.log_artifact(output_path)

    # Log individual metrics
    mlflow.log_metrics({
        "accuracy": report['accuracy'],
        "precision": report['macro avg']['precision'],
        "recall": report['macro avg']['recall'],
        "f1-score": report['macro avg']['f1-score'],
    })

    return df_report

def visualize_results(results_data: pd.DataFrame, output_path: str) -> None:
    """Visualize model results using Plotly.

    Args:
        results_data (pd.DataFrame): DataFrame containing results.
        output_path (str): Path to save the visualization.
    """
    # Reshape the DataFrame
    melted_results = results_data.melt(var_name='metric', value_name='value')

    fig = px.bar(melted_results, x='metric', y='value', title='Model Performance Metrics')
    fig.write_html(output_path)

    # Log the visualization artifact to MLflow
    mlflow.log_artifact(output_path)

