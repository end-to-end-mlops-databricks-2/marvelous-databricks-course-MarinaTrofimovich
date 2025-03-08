%pip install /Volumes/mlops_dev/mtrofimo/churn_predictor/churn_predictor-0.0.1-py3-none-any.whl
%pip install loguru

import argparse

import os

from loguru import logger

from churn_predictor.config import ProjectConfig
from churn_predictor.serving.model_serving import ModelServing

# Load project config
# Determine the environment and set the config path accordingly
#if "DATABRICKS_RUNTIME_VERSION" in os.environ:
#    config_path = "../project_config.yml"
#else:
#    config_path = os.path.abspath("project_config.yml")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

# Load project config
config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "churn_predictor-model-serving"

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.churn_predictor_model_basic",
    endpoint_name="churn_predictor-model-serving",
)

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()
