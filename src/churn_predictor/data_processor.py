import datetime

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from churn_predictor.config import ProjectConfig


class DataProcessor:
    def __init__(self, filepath: str, config: ProjectConfig):
        self.df = pd.read_csv(filepath)  # Read a csv file and store it as pandas df
        self.config = config  # Store the configuration

    def preprocess_data(self):
        """Preprocess the DataFrame stored in self.df"""
        #self.df = self.df.fillna(0)  # Fill NaN values with 0

        #remove the columns that aren't supposed to affect the churn
        self.df.drop(["Tenure", "EstimatedSalary", "HasCrCard"], axis=1, inplace=True)

        #remove the outliers
        self.df = self.df.drop(self.df[(self.df["Exited"] == 0) & (self.df["Age"] > 56)].index)
        self.df = self.df.drop(self.df[(self.df["Exited"] == 1) & (self.df["Age"] > 70)].index)
        self.df.drop(self.df[(self.df["Exited"] == 1) & (self.df["CreditScore"] < 350)].index)

        #create dummy variables for categorical features
        dummy_gend = pd.get_dummies(self.df["Gender"]).astype(int)
        dummy_gend = dummy_gend.astype(int)
        dummy_geo = pd.get_dummies(self.df["Geography"]).astype(int)
        dummy_geo = dummy_geo.astype(int)

        #add dummy columns to the dataset
        self.df = pd.concat([self.df, dummy_gend, dummy_geo], axis=1)

        #encode  "NumOfProducts" as a categorical variable
        dummy_np = pd.get_dummies(self.df["NumOfProducts"])
        dummy_np.columns = dummy_np.columns.astype(str)
        dummy_np = dummy_np.rename(columns={
                                            '1': 'NumOfProducts1',
                                            '2': 'NumOfProducts2',
                                            '3': 'NumOfProducts3',
                                            '4': 'NumOfProducts4'
                                           }
                                    )
        #add dummy column to the dataset
        self.df = pd.concat([self.df, dummy_np], axis=1)

        #drop the original columns "Gender", "Geography", "NumOfProducts"
        self.df.drop(["Gender", "Geography", "NumOfProducts"], axis=1, inplace=True)

        print(self.df.head())
 
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
        relevant_columns = cat_features + num_features + [target]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )