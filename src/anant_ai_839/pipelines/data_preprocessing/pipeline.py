"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_data,
            inputs="dataset_id_742",
            outputs="preprocessed_dataset",
            name="preprocess_dataset_node",
        ),
    ])
