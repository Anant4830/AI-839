"""
This is a boilerplate pipeline 'data_reporting'
generated using Kedro 0.19.8
"""

# from kedro.pipeline import Pipeline, pipeline


# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])

# src/anant_ai_839/pipelines/data_reporting/pipeline.py
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import generate_drift_report, evaluate_model_performance, visualize_results

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_drift_report,
                inputs=["X_train", "X_test", "params:reporting.drift_report_path"],
                outputs=None,
                name="generate_drift_report_node"
            ),
            node(
                func=evaluate_model_performance,
                inputs=["y_test", "y_pred", "params:reporting.classification_report_path"],
                outputs="model_performance_metrics",
                name="evaluate_model_performance_node"
            ),
            node(
                func=visualize_results,
                inputs=["model_performance_metrics", "params:reporting.visualization_path"],
                outputs=None,
                name="visualize_results_node"
            ),
        ]
    )
