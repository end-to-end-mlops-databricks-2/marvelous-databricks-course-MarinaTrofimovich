import datetime
import itertools
import os
import time

import requests
from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, to_utc_timestamp

from churn_predictor.config import ProjectConfig
from churn_predictor.data_processor import generate_synthetic_data_with_drift
from churn_predictor.monitoring import create_or_refresh_monitoring

# Initialize Spark Session
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Load configuration
config_path = "project_config.yml" if "DATABRICKS_RUNTIME_VERSION" not in os.environ else "../project_config.yml"
config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# Load datasets
train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

print(train_set.columns)

# Generate synthetic data
inference_data_skewed = generate_synthetic_data_with_drift(train_set, True, 200)
inference_data_skewed_spark = spark.createDataFrame(inference_data_skewed).withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)


inference_data_skewed_spark.write.mode("overwrite").saveAsTable(
    f"{config.catalog_name}.{config.schema_name}.inference_data_skewed"
)

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").withColumn(
    "CustomerId", col("CustomerId").cast("string")
)
inference_data_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed").withColumn(
    "CustomerId", col("CustomerId").cast("string")
)

print(test_set.head())
print(inference_data_skewed.head())

# Get Databricks API token and workspace URL
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# Define required columns for inference
required_columns = [
    "Geography",
    "Gender",
    "NumOfProducts",
    "CreditScore",
    "Age",
    "Balance",
    "IsActiveMember",
    "CustomerId",
]
sampled_skewed_records = inference_data_skewed[required_columns].to_dict(orient="records")
test_set_records = test_set[required_columns].to_dict(orient="records")


def send_request_https(dataframe_record):
    model_serving_endpoint = f"https://{host}/serving-endpoints/churn_predictor-model-serving/invocations"
    response = requests.post(
        model_serving_endpoint,
        headers={"Authorization": f"Bearer {token}"},
        json={"dataframe_records": [dataframe_record]},
    )
    return response


# Send requests for test records for 20 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=20)
for index, record in enumerate(itertools.cycle(test_set_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for test data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# Send requests for skewed records for 30 minutes
end_time = datetime.datetime.now() + datetime.timedelta(minutes=30)
for index, record in enumerate(itertools.cycle(sampled_skewed_records)):
    if datetime.datetime.now() >= end_time:
        break
    print(f"Sending request for skewed data, index {index}")
    response = send_request_https(record)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    time.sleep(0.2)

# Refresh monitoring
workspace = WorkspaceClient()
create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)
