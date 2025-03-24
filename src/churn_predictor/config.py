from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters
    pipeline_id: str  # pipeline id for data live tables
    experiment_name: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = None):
        """Load configuration from a YAML file, adapting based on environment settings."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if env is not None:
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
        else:
            config_dict["catalog_name"] = config_dict["catalog_name"]
            config_dict["schema_name"] = config_dict["schema_name"]

        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
    job_run_id: str
