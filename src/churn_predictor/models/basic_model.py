import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churn_predictor.config import ProjectConfig, Tags

"""
infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
"""

"""
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""


class BasicModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        """
        Initialize the model with project configuration.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name
        self.tags = tags.dict()

    def load_data(self):
        """
        Load training and testing data from Delta tables.
        Splits data into:
        Features (X_train, X_test)
        Target (y_train, y_test)
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self):
        """
        Encodes categorical features with OneHotEncoder (ignores unseen categories).
        Passes numerical features as-is (remainder='passthrough').
        Defines a pipeline combining:
            Features processing
            LightGBM regression model
        """
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self):
        """
        Train the model.
        """
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self):
        """
        Log the model.
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline, artifact_path="lightgbm-pipeline-model", signature=signature
            )

    def register_model(self):
        """
        Register model in UC
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=f"{self.catalog_name}.{self.schema_name}.churn_predictor_model_basic",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.churn_predictor_model_basic",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self):
        """
        Retrieve MLflow run dataset.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        """
        Retrieve MLflow run metadata.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame):
        """
        Load the latest model from MLflow (alias=latest-model) and make predictions.
        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.churn_predictor_model_basic@latest-model"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions

    def model_improved(self, test_set):
        """
        Evaluate the model performance on the test set.
        """

        test_set_wot = test_set.drop(self.config.target)
        X_test_sp = test_set_wot.select("*")
        X_test_sp_wt = test_set.select("*")
        X_test_sp.display()

        X_test = X_test_sp.toPandas()
        X_test_wt = X_test_sp_wt.toPandas()

        predictions_latest = self.load_latest_model_and_predict(X_test)
        predictions_latest_df = pd.DataFrame(predictions_latest, columns=["prediction"])
        predictions_latest_df = predictions_latest_df.rename(columns={"prediction": "prediction_latest"})
        predictions_latest_df["CustomerId"] = X_test_wt["CustomerId"]
        logger.info("predictions_latest")
        logger.info(predictions_latest_df)
        predictions_latest_df.display()

        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model"
        current_model = mlflow.sklearn.load_model(current_model_uri)
        logger.info("✅ Model successfully loaded.")
        predictions_current = current_model.predict(X_test)
        predictions_current_df = pd.DataFrame(predictions_current, columns=["prediction"])
        predictions_current_df = predictions_current_df.rename(columns={"prediction": "prediction_current"})
        predictions_current_df["CustomerId"] = X_test_wt["CustomerId"]
        logger.info("predictions_current")
        logger.info(predictions_current)
        predictions_current_df.display()

        X_test_wt = X_test_wt.loc[:, ["CustomerId", "Exited"]]

        logger.info("Predictions are ready.")

        df = X_test_wt.merge(predictions_current_df, on="CustomerId").merge(predictions_latest_df, on="CustomerId")

        # Calculate the recall for each model
        recall_current = recall_score(df["Exited"], df["prediction_current"])
        recall_latest = recall_score(df["Exited"], df["prediction_latest"])

        # Log the recall scores
        logger.info(f"Recall for Current Model: {recall_current}")
        logger.info(f"Recall for Latest Model: {recall_latest}")

        # Compare models based on recall
        if recall_current > recall_latest:
            logger.info("Current Model performs better. Registering new model.")
            return True
        else:
            logger.info("New Model performs worse. Keeping the old model.")
            return False