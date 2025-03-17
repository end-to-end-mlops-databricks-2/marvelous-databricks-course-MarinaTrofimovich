{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "application/vnd.databricks.v1+cell": {
     "cellMetadata": {},
     "inputWidgets": {},
     "nuid": "d41c9efd-03bb-4ec4-b7a8-942388aa9492",
     "showTitle": false,
     "tableResultSettingsMap": {},
     "title": ""
    }
   },
   "outputs": [],
   "source": [
    "import argparse\n",
    "\n",
    "from databricks.connect import DatabricksSession\n",
    "from databricks.sdk import WorkspaceClient\n",
    "from house_price.config import ProjectConfig\n",
    "from house_price.monitoring import create_or_refresh_monitoring\n",
    "\n",
    "parser = argparse.ArgumentParser()\n",
    "parser.add_argument(\n",
    "    \"--root_path\",\n",
    "    action=\"store\",\n",
    "    default=None,\n",
    "    type=str,\n",
    "    required=True,\n",
    ")\n",
    "\n",
    "parser.add_argument(\n",
    "    \"--env\",\n",
    "    action=\"store\",\n",
    "    default=None,\n",
    "    type=str,\n",
    "    required=True,\n",
    ")\n",
    "\n",
    "args = parser.parse_args()\n",
    "root_path = args.root_path\n",
    "config_path = f\"{root_path}/files/project_config.yml\"\n",
    "\n",
    "# Load configuration\n",
    "config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)\n",
    "\n",
    "spark = DatabricksSession.builder.getOrCreate()\n",
    "workspace = WorkspaceClient()\n",
    "\n",
    "create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)"
   ]
  }
 ],
 "metadata": {
  "application/vnd.databricks.v1+notebook": {
   "computePreferences": null,
   "dashboards": [],
   "environmentMetadata": {
    "base_environment": "",
    "environment_version": "2"
   },
   "language": "python",
   "notebookMetadata": {
    "pythonIndentUnit": 4
   },
   "notebookName": "04_refresh_monitor.py",
   "widgets": {}
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
