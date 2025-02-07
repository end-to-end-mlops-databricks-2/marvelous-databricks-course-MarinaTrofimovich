#%pip install /Volumes/mlops_dev/mtrofimo/churn_predictor/churn_predictor-0.0.1-py3-none-any.whl
%pip install loguru
from loguru import logger

import os

import yaml
from pyspark.sql import SparkSession

from churn_predictor.config import ProjectConfig
from churn_predictor.data_processor import DataProcessor


# Determine the environment and set the config path accordingly
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    config_path = "../project_config.yml"
else:
    config_path = os.path.abspath("project_config.yml")

print("config_path:", config_path)
config = ProjectConfig.from_yaml(config_path=config_path)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# Load the house prices dataset
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/data.csv", header=True, inferSchema=True
).toPandas()

df.display()

# COMMAND ----------
# Initialize DataProcessor

#if "DATABRICKS_RUNTIME_VERSION" in os.environ:
#    data_path = "../data/data.csv"
#else:
#    data_path = os.path.abspath("data/data.csv")

# Initialize DataProcessor
data_processor = DataProcessor(df, config, spark)

# Preprocess the data
data_processor.preprocess_data()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
