#%pip install /Volumes/mlops_dev/mtrofimo/churn_predictor/churn_predictor-0.0.1-py3-none-any.whl
#%pip install loguru

import mlflow
import os
from pyspark.sql import SparkSession

from churn_predictor.config import ProjectConfig, Tags
from churn_predictor.models.basic_model import BasicModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

#config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# Determine the environment and set the config path accordingly
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    config_path = "../project_config.yml"
else:
    config_path = os.path.abspath("project_config.yml")

config = ProjectConfig.from_yaml(config_path=config_path)

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)

# Initialize model
basic_model = BasicModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
basic_model.train()
basic_model.log_model()

# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/churn-predictor-basic"], filter_string="tags.branch='week2'"
).run_id[0]

model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()

predictions_df = basic_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
