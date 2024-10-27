"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            func=split_data,
            inputs=["dataset_id_742", "params:model_options"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=preprocess_data,
            inputs=["X_train", "X_test", "y_train", "params:scaler_path", "params:columns_path"],
            outputs=["X_train_encoded", "X_test_encoded", "y_train_int"],
            name="preprocess_dataset_node",
        ),
    ])
