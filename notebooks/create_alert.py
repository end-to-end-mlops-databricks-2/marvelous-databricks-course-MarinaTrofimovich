import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()

# Define the alert query for classification model
alert_query = """
SELECT
  (COUNT(CASE WHEN accuracy_score < 0.85 THEN 1 END) * 100.0 / COUNT(CASE WHEN accuracy_score IS NOT NULL AND NOT isnan(accuracy_score) THEN 1 END)) AS percentage_lower_than_85
FROM mlops_dev.mtrofimo.model_monitoring_profile_metrics"""

# Create the query in Databricks
query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"classification-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on classification model accuracy",
        query_text=alert_query,
    )
)

# Create the alert based on the query
alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="percentage_lower_than_85")),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=20)),
        ),
        display_name=f"classification-accuracy-alert-{time.time_ns()}",
        query_id=query.id,
    )
)

# COMMAND ----------

# Cleanup
# w.queries.delete(id=query.id)
# w.alerts.delete(id=alert.id)
