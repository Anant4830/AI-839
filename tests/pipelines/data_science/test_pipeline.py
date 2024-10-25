"""
This is a boilerplate test file for pipeline 'data_science'
generated using Kedro 0.19.8.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

# NOTE: This example test is yet to be refactored.
# A complete version is available under the testing best practices section.

import logging
import pandas as pd
import pytest

from kedro.io import DataCatalog
from kedro.runner import SequentialRunner
from anant_ai_839.pipelines.data_science import create_pipeline as create_ds_pipeline
from anant_ai_839.pipelines.data_science.nodes import split_data

@pytest.fixture
def dummy_data():
    return pd.DataFrame(
        {
            "age": [18, 20, 33],
            "credit_amount": [4000, 5000, 60000],
            "y": [1, 0, 0],
        }
    )

@pytest.fixture
def dummy_parameters():
    parameters = {
        "model_options": {
            "test_size": 0.2,
            "random_state": 3,
            "features": ["age", "credit_amount"],
        }
    }
    return parameters


def test_split_data(dummy_data, dummy_parameters):
    X_train, X_test, y_train, y_test = split_data(
        dummy_data, dummy_parameters["model_options"]
    )
    assert len(X_train) == 2
    assert len(y_train) == 2
    assert len(X_test) == 1
    assert len(y_test) == 1

def test_split_data_missing_price(dummy_data, dummy_parameters):
    dummy_data_missing_y = dummy_data.drop(columns="y")
    with pytest.raises(KeyError) as e_info:
        X_train, X_test, y_train, y_test = split_data(dummy_data_missing_y, dummy_parameters["model_options"])

    assert "y" in str(e_info.value)

def test_data_science_pipeline(caplog, dummy_data, dummy_parameters):
    pipeline = (
        create_ds_pipeline()
        .from_nodes("split_data_node")
        .to_nodes("evaluate_model_node")
    )
    catalog = DataCatalog()
    catalog.add_feed_dict(
        {
            "preprocessed_dataset" : dummy_data,
            "params:model_options": dummy_parameters["model_options"],
        }
    )

    caplog.set_level(logging.DEBUG, logger="kedro")
    successful_run_msg = "Pipeline execution completed successfully."

    SequentialRunner().run(pipeline, catalog)

    assert successful_run_msg in caplog.text
