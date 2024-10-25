from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession

import mlflow
import logging
import typing as t

logger = logging.getLogger(__name__)

class ExtraMLflowHooks:
    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, t.Any]):
        logger.info("Logging extra metadata to MLflow")
        mlflow.set_tags({
            "pipeline": run_params["pipeline_name"] or "__default__",
            "custom_version": "0.1.0",
        })


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        """Initialises a SparkSession using the config
        defined in project's conf folder.
        """

        # Load the spark configuration in spark.yaml using the config loader
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # Initialise the spark session
        spark_session_conf = (
            SparkSession.builder.appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
        )
        _spark_session = spark_session_conf.getOrCreate()
        _spark_session.sparkContext.setLogLevel("WARN")
