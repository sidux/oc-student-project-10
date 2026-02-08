terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.0.0, < 4.0.0"
    }
  }
}

# Azure Provider configuration
provider "azurerm" {
  features {}
}

# Define variables to avoid string duplication
variable "resource_group_name" {
  description = "Name of the resource group"
  default     = "oc-student-recommendation"
}

variable "location" {
  description = "Azure region where resources will be created"
  default     = "West Europe"
}

variable "storage_account_name" {
  description = "Name of the storage account"
  default     = "ocrecommendationstorage"
}

variable "app_name_prefix" {
  description = "Prefix for application resource names"
  default     = "oc-recommendation"
}

# Create a new resource group specifically for this project
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
}

# Storage Account for both Azure Functions and storing models
resource "azurerm_storage_account" "main" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  # Allow Blob public access
  allow_nested_items_to_be_public = true
}

# Blob Container specifically for storing ML models
resource "azurerm_storage_container" "models" {
  name                  = "models"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private" # Secure access to models
}

# Consumption Plan for Azure Functions (serverless)
resource "azurerm_service_plan" "main" {
  name                = "${var.app_name_prefix}-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "EP2" # Consumption plan (serverless)
}

# Function App
resource "azurerm_linux_function_app" "main" {
  name                = "${var.app_name_prefix}-app"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  storage_account_name       = azurerm_storage_account.main.name
  storage_account_access_key = azurerm_storage_account.main.primary_access_key
  service_plan_id            = azurerm_service_plan.main.id

  site_config {
    application_stack {
      python_version = "3.11"
    }
    application_insights_key = azurerm_application_insights.main.instrumentation_key
  }

  app_settings = {
    "WEBSITE_RUN_FROM_PACKAGE"           = "1",
    "FUNCTIONS_WORKER_RUNTIME"           = "python",
    "AZURE_STORAGE_CONNECTION_STRING"    = azurerm_storage_account.main.primary_connection_string,
    "MODELS_CONTAINER_NAME"              = azurerm_storage_container.models.name,
    "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
  }

  # Increase function timeout and resources for ML workloads
  identity {
    type = "SystemAssigned"
  }
}

# Application Insights for monitoring
resource "azurerm_application_insights" "main" {
  name                = "${var.app_name_prefix}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
}

# Outputs
output "function_app_name" {
  value = azurerm_linux_function_app.main.name
}

output "function_app_default_hostname" {
  value = azurerm_linux_function_app.main.default_hostname
}

output "storage_account_name" {
  value = azurerm_storage_account.main.name
}

output "storage_account_connection_string" {
  value     = azurerm_storage_account.main.primary_connection_string
  sensitive = true
}

output "models_container_name" {
  value = azurerm_storage_container.models.name
}

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}