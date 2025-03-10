import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from churn_predictor.config import ProjectConfig


class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        # self.df = pd.read_csv(filepath)  # Read a csv file and store it as pandas df
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess_data(self):
        print("In preprocess_data.")

        """Preprocess the DataFrame stored in self.df"""
        # self.df = self.df.fillna(0)  # Fill NaN values with 0

        # remove the columns that aren't supposed to affect the churn
        self.df.drop(["Tenure", "EstimatedSalary", "HasCrCard"], axis=1, inplace=True)

        # remove the outliers
        self.df = self.df.drop(self.df[(self.df["Exited"] == 0) & (self.df["Age"] > 56)].index)
        self.df = self.df.drop(self.df[(self.df["Exited"] == 1) & (self.df["Age"] > 70)].index)
        self.df.drop(self.df[(self.df["Exited"] == 1) & (self.df["CreditScore"] < 350)].index)

        # Initialize LabelEncoder
        # le_geography = LabelEncoder()
        # le_gender = LabelEncoder()
        # Fit and transform the categorical features
        # self.df["Geography"] = le_geography.fit_transform(self.df["Geography"])
        # self.df["Gender"] = le_gender.fit_transform(self.df["Gender"])

        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Convert categorical features to the appropriate type
        cat_features = self.config.cat_features
        for cat_col in cat_features:
            self.df[cat_col] = self.df[cat_col].astype("category")

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target] + ["CustomerId"]
        self.df = self.df[relevant_columns]
        self.df.display()

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self):
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df, num_rows=10):
    """Generates synthetic data based on the distribution of the input DataFrame."""
    synthetic_data = pd.DataFrame(columns=df.columns)

    for column in df.columns:
        if column == "CustomerId" or column == "RowNumber":
            continue

        if column == "Exited":
            synthetic_data[column] = np.random.choice([0, 1], num_rows)

        elif pd.api.types.is_numeric_dtype(df[column]):
            synthetic_data[column] = np.abs(np.random.normal(df[column].mean(), df[column].std(), num_rows))

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to integers
    int_columns = {"CreditScore", "Age", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"}
    for col in int_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.int32)

    synthetic_data["EstimatedSalary"] = synthetic_data["EstimatedSalary"].astype(np.float64)

    if "Geography" in df.columns:
        synthetic_data["Geography"] = synthetic_data["Geography"].astype(df["Geography"].dtype)

    # Find the maximum RowNumber in the input DataFrame
    max_row_number = df["RowNumber"].max() if "RowNumber" in df.columns else 0

    # Create the RowNumber column starting from the next integer after the max RowNumber in df
    synthetic_data["RowNumber"] = range(max_row_number + 1, max_row_number + 1 + num_rows)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["CustomerId"] = [str(timestamp_base + i) for i in range(num_rows)]

    return synthetic_data
