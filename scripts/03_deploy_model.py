from loguru import logger

from house_price.config import ProjectConfig
from house_price.serving.fe_model_serving import FeatureLookupServing

# Load project config
# Determine the environment and set the config path accordingly
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    config_path = "../project_config.yml"
else:
    config_path = os.path.abspath("project_config.yml")

config = ProjectConfig.from_yaml(config_path=config_path)

logger.info("Loaded config file.")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "churn_predictor-model-serving"

# Initialize feature store manager
model_serving = ModelServing(model_name=f"{catalog_name}.{schema_name}.churn_predictor_model_basic",
                             endpoint_name="churn_predictor-model-serving")

# COMMAND ----------
# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()

