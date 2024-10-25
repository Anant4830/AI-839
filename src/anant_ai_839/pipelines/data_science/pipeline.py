"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, train_model, evaluate_model, plot_confusion_matrix

# def create_pipeline(**kwargs) -> Pipeline:
#     pipeline_instance = pipeline([
#         node(
#            func = split_data,
#            inputs = ["preprocessed_dataset", "params:model_options"],
#            outputs = ["X_train", "X_test", "y_train", "y_test"],
#         ),
#         node(
#             func = train_model,
#             inputs = ["X_train", "y_train"],
#             outputs = "classification",
#         ),
#         node(
#             func = evaluate_model,
#             inputs = ["classification", "X_test", "y_test"],
#             outputs = None,
#         ),
#         node(
#             func = plot_confusion_matrix,
#             inputs=["y_true", "y_pred", "params:output_path"],  # 'params:output_path' will be the path to save the confusion matrix
#             outputs=None,  # No need to output since the plot is saved to file
#             name="plot_confusion_matrix_node",
#             ),
#     ])
#     ds_pipeline_1 = pipeline(
#         pipe = pipeline_instance,
#         inputs = "preprocessed_dataset",
#         namespace = "active_modelling_pipeline",
#     )
#     ds_pipeline_2 = pipeline(
#         pipe = pipeline_instance,
#         inputs = "preprocessed_dataset",
#         namespace = "candidate_modelling_pipeline"
#     )

#     return ds_pipeline_1 + ds_pipeline_2

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a Kedro pipeline for data science tasks including data splitting,
    model training, model evaluation, and confusion matrix plotting.
    
    Returns:
        Pipeline: A configured Kedro pipeline with nodes for data science tasks.
    """
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_dataset", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="classification",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["classification", "X_test", "y_test"],
                outputs="y_pred",
                name="evaluate_model_node",
            ),
            node(
                func = plot_confusion_matrix,
                inputs=["y_test", "y_pred", "params:output_path"], 
                outputs="confusion_matrix", 
                name="plot_confusion_matrix_node",
            ),
        ]
    )
