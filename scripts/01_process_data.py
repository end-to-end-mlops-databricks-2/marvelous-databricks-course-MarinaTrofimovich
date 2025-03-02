# %pip install /Volumes/mlops_dev/mtrofimo/churn_predictor/churn_predictor-0.0.1-py3-none-any.whl
# %pip install loguru

import argparse

import os

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from churn_predictor.config import ProjectConfig
from churn_predictor.data_processor import DataProcessor, generate_synthetic_data

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

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

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

# Generate synthetic data
### This is mimicking a new data arrival. In real world, this would be a new batch of data.
# df is passed to infer schema
synthetic_df = generate_synthetic_data(df, num_rows=100)
synthetic_df.display()
logger.info("Synthetic data generated.")

# Initialize DataProcessor
data_processor = DataProcessor(synthetic_df, config, spark)

# Preprocess the data
data_processor.preprocess_data()

# Split the data
X_train, X_test = data_processor.split_data()
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)
