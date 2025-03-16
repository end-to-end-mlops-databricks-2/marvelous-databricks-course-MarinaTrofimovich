from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from loguru import logger
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from databricks.sdk.errors import NotFound


def create_or_refresh_monitoring(config, spark, workspace):

    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`churn_predictor-model-serving_payload_payload`")

    request_schema = StructType([
        StructField("dataframe_records", ArrayType(StructType([
            StructField("Geography", StringType(), True),
            StructField("Gender", StringType(), True),
            StructField("NumOfProducts", IntegerType(), True),
            StructField("CreditScore", IntegerType(), True),
            StructField("Age", IntegerType(), True),
            StructField("Balance", DoubleType(), True),
            StructField("IsActiveMember", IntegerType(), True),
            StructField("CustomerId", StringType(), True)
        ])), True)  
    ])

    response_schema = StructType([
        StructField("predictions", ArrayType(DoubleType()), True),
        StructField("databricks_output", StructType([
            StructField("trace", StringType(), True),
            StructField("databricks_request_id", StringType(), True)
        ]), True)
    ])

    
    # Check the content of the request column
    inf_table.select("request").show(truncate=False)

    inf_table_parsed = inf_table.withColumn("parsed_request", 
                                            F.from_json(F.col("request"),
                                                        request_schema))
    
    # Check the content of the parsed_request column
    inf_table_parsed.select("parsed_request").show(truncate=False)

    inf_table_parsed.display()

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response",
                                                F.from_json(F.col("response"),
                                                            response_schema))
    # Check the content of the parsed_response column
    inf_table_parsed.select("parsed_response").show(truncate=False)
    inf_table_parsed.display()

    df_exploded = inf_table_parsed.withColumn("record",
                                            F.explode(F.col("parsed_request.dataframe_records")))

    df_exploded.display()


    df_final = df_exploded.select(
        F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        F.col("record.CustomerId").alias("CustomerId"),
        F.col("record.Geography").alias("Geography"),
        F.col("record.CreditScore").alias("CreditScore"),
        F.col("record.Age").alias("Age"),
        F.col("record.Balance").alias("Balance"),
        F.col("record.IsActiveMember").alias("IsActiveMember"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("churn-predictor-model-fe").alias("model_name")
    )


    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    test_set.display()
    inference_set_skewed.display()


    df_final_with_status = df_final \
        .join(test_set.select("CustomerId", "Exited"), on="CustomerId", how="left") \
        .withColumnRenamed("Exited", "Exited_test") \
        .join(inference_set_skewed.select("CustomerId", "Exited"), on="CustomerId", how="left") \
        .withColumnRenamed("Exited", "Exited_inference") \
        .select(
            "*",  
            F.coalesce(F.col("Exited_test"), F.col("Exited_inference")).alias("Exited")
        ) \
        .drop("Exited_test", "Exited_inference") \
        .withColumn("Exited", F.col("Exited").cast("double")) \
        .withColumn("prediction", F.col("prediction").cast("double")) \
        .dropna(subset=["Exited", "prediction"])

    df_final_with_status.write.format("delta").mode("append")\
        .saveAsTable(f"{config.catalog_name}.{config.schema_name}.model_monitoring")

    
    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")



def create_monitoring_table(config, spark, workspace):
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"], 
            model_id_col="model_name",
            label_col="sale_price",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")