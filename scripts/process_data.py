%pip install /Volumes/mlops_dev/mtrofimo/churn_predictor/churn_predictor-0.0.1-py3-none-any.whl

import yaml
import os

from churn_predictor.data_processor import DataProcessor
from churn_predictor.config import ProjectConfig

# Load configuration
config_path = os.path.abspath("project_config.yml")
print('config_path:', config_path)
config = ProjectConfig.from_yaml(config_path=config_path)
#config = ProjectConfig.from_yaml(config_path="../project_config.yml")

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
data_path = os.path.abspath("data/data.csv")
data_processor = DataProcessor(data_path, config)
#data_processor = DataProcessor("../data/data.csv", config)

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------
# Split the data
X_train, X_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print(X_train.head())
print(X_test.head())    

#data_processor.save_to_catalog(X_train, X_test)