"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import load_model, predict, log_inference_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=load_model,
                inputs="params:model_path_to_load",
                outputs="loaded_model",
                name="load_model_node"
            ),
            node(
                func=predict,
                inputs=["loaded_model", "input_features"],
                outputs="predictions",
                name="predict_node"
            ),
            node(
                func=log_inference_data,
                inputs=["params:model_name", "input_features", "predictions"],
                outputs=None,
                name="log_inference_data_node"
            ),
    ])
