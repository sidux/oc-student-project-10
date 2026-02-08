"""
Pulumi Infrastructure for Recommendation System
Deploys Azure Functions with Storage Account for ML models
"""

import re

from pulumi import Output, export, get_project
from pulumi_azure_native import applicationinsights, operationalinsights, resources, storage, web

LOCATION = "francecentral"
PROJECT = get_project()


def stack_name(suffix: str) -> str:
    return f"{PROJECT}-{suffix}"


def sanitize_storage_name(raw: str) -> str:
    """Azure Storage Account names must be 3-24 lowercase alphanumeric characters."""
    clean = re.sub("[^a-z0-9]", "", raw.lower())
    if len(clean) > 24:
        clean = clean[:24]
    if len(clean) < 3:
        raise ValueError("Storage name must be at least 3 characters.")
    return clean


STORAGE_NAME = sanitize_storage_name(PROJECT)

# Resource Group
resource_group = resources.ResourceGroup(
    "resource_group",
    resource_group_name=PROJECT,
    location=LOCATION,
)

# Storage Account for Functions and ML models
storage_account = storage.StorageAccount(
    "storage_account",
    account_name=STORAGE_NAME,
    resource_group_name=resource_group.name,
    location=resource_group.location,
    sku=storage.SkuArgs(name=storage.SkuName.STANDARD_LRS),
    kind=storage.Kind.STORAGE_V2,
    allow_blob_public_access=False,
)

# Blob Container for ML models
models_container = storage.BlobContainer(
    "models_container",
    container_name="models",
    account_name=storage_account.name,
    resource_group_name=resource_group.name,
    public_access=storage.PublicAccess.NONE,
)

# Get storage account keys
storage_keys = storage.list_storage_account_keys_output(
    account_name=storage_account.name,
    resource_group_name=resource_group.name,
)
primary_storage_key = storage_keys.keys[0].value
storage_connection_string = Output.concat(
    "DefaultEndpointsProtocol=https;AccountName=",
    storage_account.name,
    ";AccountKey=",
    primary_storage_key,
    ";EndpointSuffix=core.windows.net",
)

# Log Analytics Workspace (required for Application Insights)
log_analytics = operationalinsights.Workspace(
    "log_analytics",
    workspace_name=stack_name("logs"),
    resource_group_name=resource_group.name,
    location=resource_group.location,
    sku=operationalinsights.WorkspaceSkuArgs(
        name=operationalinsights.WorkspaceSkuNameEnum.PER_GB2018,
    ),
    retention_in_days=30,
)

# Application Insights for monitoring
app_insights = applicationinsights.Component(
    "app_insights",
    resource_name_=stack_name("insights"),
    resource_group_name=resource_group.name,
    location=resource_group.location,
    kind="web",
    application_type=applicationinsights.ApplicationType.WEB,
    workspace_resource_id=log_analytics.id,
    ingestion_mode=applicationinsights.IngestionMode.LOG_ANALYTICS,
)

# App Service Plan (Consumption for serverless)
service_plan = web.AppServicePlan(
    "service_plan",
    name=stack_name("plan"),
    location=resource_group.location,
    resource_group_name=resource_group.name,
    kind="Linux",
    reserved=True,
    sku=web.SkuDescriptionArgs(
        name="Y1",
        tier="Dynamic",
    ),
)

# Linux Function App
function_app = web.WebApp(
    "function_app",
    name=stack_name("func"),
    resource_group_name=resource_group.name,
    location=resource_group.location,
    server_farm_id=service_plan.id,
    kind="functionapp,linux",
    site_config=web.SiteConfigArgs(
        linux_fx_version="PYTHON|3.11",
        app_settings=[
            web.NameValuePairArgs(name="AzureWebJobsStorage", value=storage_connection_string),
            web.NameValuePairArgs(name="FUNCTIONS_EXTENSION_VERSION", value="~4"),
            web.NameValuePairArgs(name="FUNCTIONS_WORKER_RUNTIME", value="python"),
            web.NameValuePairArgs(name="WEBSITE_RUN_FROM_PACKAGE", value="1"),
            web.NameValuePairArgs(name="AZURE_STORAGE_CONNECTION_STRING", value=storage_connection_string),
            web.NameValuePairArgs(name="MODELS_CONTAINER_NAME", value="models"),
            web.NameValuePairArgs(name="APPINSIGHTS_INSTRUMENTATIONKEY", value=app_insights.instrumentation_key),
            web.NameValuePairArgs(name="APPLICATIONINSIGHTS_CONNECTION_STRING", value=app_insights.connection_string),
        ],
    ),
)

# Exports
export("function_app_name", function_app.name)
export("function_app_url", Output.concat("https://", function_app.default_host_name))
export("storage_account_name", storage_account.name)
export("storage_connection_string", Output.secret(storage_connection_string))
export("models_container_name", models_container.name)
export("app_insights_key", Output.secret(app_insights.instrumentation_key))
export("resource_group_name", resource_group.name)
