#!/usr/bin/env bash
set -euo pipefail

# Recommendation System - Azure Functions Deployment Script

STACK="${1:-dev}"
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─────────────────────────────────────────────
# Colors and Formatting
# ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

success() { echo -e "${GREEN}✓${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
info() { echo -e "${BLUE}→${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; }

echo ""
echo "=========================================="
echo "  Recommendation System - Azure Deploy"
echo "=========================================="
echo ""

# ─────────────────────────────────────────────
# Prerequisites Check
# ─────────────────────────────────────────────
header "1. Checking Prerequisites"

MISSING_DEPS=0

# Check command exists and show version
check_command() {
  local cmd="$1"
  local install_hint="$2"
  local version_flag="${3:---version}"
  
  if command -v "$cmd" &>/dev/null; then
    local version
    version=$("$cmd" $version_flag 2>&1 | head -1 || echo "installed")
    success "$cmd: $version"
  else
    error "$cmd: NOT INSTALLED"
    echo "      Install: $install_hint"
    MISSING_DEPS=1
  fi
}

# Required tools (disable exit on error for checks)
set +e
check_command "az" "brew install azure-cli OR https://docs.microsoft.com/cli/azure/install-azure-cli" "--version"
check_command "func" "npm install -g azure-functions-core-tools@4" "--version"
check_command "pulumi" "brew install pulumi OR https://www.pulumi.com/docs/install/" "version"
check_command "python3" "brew install python@3.11" "--version"
check_command "uv" "curl -LsSf https://astral.sh/uv/install.sh | sh" "--version"
check_command "curl" "brew install curl" "--version"

# Optional but recommended
if command -v npm &>/dev/null; then
  success "npm: $(npm --version 2>/dev/null || echo 'installed') (optional)"
else
  warn "npm: not installed (needed if func is missing)"
fi
set -e

if [[ $MISSING_DEPS -eq 1 ]]; then
  echo ""
  error "Missing required dependencies. Please install them and retry."
  exit 1
fi

# ─────────────────────────────────────────────
# Azure Login Check
# ─────────────────────────────────────────────
header "2. Checking Azure Login"

if az account show &>/dev/null; then
  SUBSCRIPTION=$(az account show --query name -o tsv)
  SUBSCRIPTION_ID=$(az account show --query id -o tsv)
  success "Logged into Azure"
  info "Subscription: $SUBSCRIPTION"
  info "ID: $SUBSCRIPTION_ID"
else
  warn "Not logged into Azure"
  info "Running 'az login'..."
  echo ""
  az login
  
  if ! az account show &>/dev/null; then
    error "Azure login failed"
    exit 1
  fi
  success "Azure login successful"
fi

# ─────────────────────────────────────────────
# Pulumi Login Check
# ─────────────────────────────────────────────
header "3. Checking Pulumi Login"

if pulumi whoami &>/dev/null; then
  PULUMI_USER=$(pulumi whoami)
  PULUMI_BACKEND=$(pulumi whoami -v 2>&1 | grep -i "backend" | head -1 || echo "")
  success "Logged into Pulumi as: $PULUMI_USER"
  if [[ -n "$PULUMI_BACKEND" ]]; then
    info "$PULUMI_BACKEND"
  fi
else
  warn "Not logged into Pulumi"
  info "Running 'pulumi login'..."
  echo ""
  pulumi login
  
  if ! pulumi whoami &>/dev/null; then
    error "Pulumi login failed"
    exit 1
  fi
  success "Pulumi login successful"
fi

# ─────────────────────────────────────────────
# Check Pulumi Project
# ─────────────────────────────────────────────
header "4. Checking Project Configuration"

if [[ ! -f "$ROOT_DIR/infra/Pulumi.yaml" ]]; then
  error "Pulumi.yaml not found in infra/"
  exit 1
fi

PULUMI_PROJECT="$(awk '/^name:/ {print $2; exit}' "$ROOT_DIR/infra/Pulumi.yaml")"
success "Pulumi project: $PULUMI_PROJECT"
info "Stack: $STACK"

# ─────────────────────────────────────────────
# Check Model Files
# ─────────────────────────────────────────────
header "5. Checking Model Files"

REQUIRED_MODELS=(
  "svd_model.pkl"
  "article_embeddings.pkl"
  "article_id_mapping.pkl"
  "user_article_ratings.csv"
  "article_popularity.csv"
)

MODELS_FOUND=0
MODELS_MISSING=0

for model in "${REQUIRED_MODELS[@]}"; do
  if [[ -f "$ROOT_DIR/models/$model" ]]; then
    size=$(du -h "$ROOT_DIR/models/$model" 2>/dev/null | cut -f1 || echo "?")
    success "$model ($size)"
    MODELS_FOUND=$((MODELS_FOUND + 1))
  else
    warn "$model: MISSING"
    MODELS_MISSING=$((MODELS_MISSING + 1))
  fi
done

if [[ $MODELS_MISSING -gt 0 ]]; then
  warn "$MODELS_MISSING model file(s) missing - API will use fallback recommendations"
  echo ""
  read -p "Continue anyway? [y/N] " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    info "Run the notebook to generate models first"
    exit 1
  fi
else
  success "All required models present"
fi

# ─────────────────────────────────────────────
# Install Python Dependencies
# ─────────────────────────────────────────────
header "6. Installing Dependencies"

info "Installing Pulumi Python dependencies..."
uv pip install -r "$ROOT_DIR/infra/requirements.txt" --quiet
success "Pulumi dependencies installed"

# ─────────────────────────────────────────────
# Deploy Infrastructure
# ─────────────────────────────────────────────
header "7. Deploying Infrastructure"

cd "$ROOT_DIR/infra"

if ! pulumi stack select "$STACK" &>/dev/null 2>&1; then
  info "Creating new Pulumi stack: $STACK"
  pulumi stack init "$STACK"
fi

info "Running 'pulumi up' (this may take a few minutes)..."
echo ""
pulumi up --stack "$STACK" --yes

# Get outputs
FUNC_NAME=$(pulumi stack output --stack "$STACK" function_app_name)
FUNC_URL=$(pulumi stack output --stack "$STACK" function_app_url)
STORAGE_ACCOUNT=$(pulumi stack output --stack "$STACK" storage_account_name)
RESOURCE_GROUP=$(pulumi stack output --stack "$STACK" resource_group_name)

echo ""
success "Infrastructure deployed"
info "Function App: $FUNC_NAME"
info "Storage Account: $STORAGE_ACCOUNT"
info "Resource Group: $RESOURCE_GROUP"

cd "$ROOT_DIR"

# ─────────────────────────────────────────────
# Upload Models
# ─────────────────────────────────────────────
header "8. Uploading Models to Azure Storage"

UPLOADED=0
SKIPPED=0

# Find model files
shopt -s nullglob
MODEL_FILES=("$ROOT_DIR"/models/*.pkl "$ROOT_DIR"/models/*.csv "$ROOT_DIR"/models/*.joblib)
shopt -u nullglob

if [[ ${#MODEL_FILES[@]} -gt 0 ]]; then
  # Get storage connection string for upload
  info "Getting storage connection string..."
  STORAGE_CONNECTION=$(az storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query connectionString -o tsv 2>/dev/null) || STORAGE_CONNECTION=""
  
  if [[ -z "$STORAGE_CONNECTION" ]]; then
    warn "Could not get storage connection string, trying with login auth..."
  fi
  
  for local_file in "${MODEL_FILES[@]}"; do
    [[ ! -f "$local_file" ]] && continue
    blob_name=$(basename "$local_file")
    filesize=$(du -h "$local_file" | cut -f1)
    
    info "Uploading $blob_name ($filesize)..."
    
    if [[ -n "$STORAGE_CONNECTION" ]]; then
      # Use connection string
      if az storage blob upload \
        --connection-string "$STORAGE_CONNECTION" \
        --container-name "models" \
        --name "$blob_name" \
        --file "$local_file" \
        --overwrite \
        --only-show-errors; then
        success "$blob_name uploaded"
        UPLOADED=$((UPLOADED + 1))
      else
        warn "Failed to upload $blob_name"
      fi
    else
      # Fallback to login auth
      if az storage blob upload \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "models" \
        --name "$blob_name" \
        --file "$local_file" \
        --overwrite \
        --auth-mode login; then
        success "$blob_name uploaded"
        UPLOADED=$((UPLOADED + 1))
      else
        warn "Failed to upload $blob_name"
      fi
    fi
  done
  
  echo ""
  success "Models: $UPLOADED files uploaded"
else
  warn "No model files found in models/ directory"
fi

# ─────────────────────────────────────────────
# Deploy Azure Functions
# ─────────────────────────────────────────────
header "9. Deploying Azure Functions Code"

cd "$ROOT_DIR/app"
info "Publishing to $FUNC_NAME..."
echo ""
func azure functionapp publish "$FUNC_NAME" --python
cd "$ROOT_DIR"

echo ""
success "Azure Functions deployed"

# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────
header "10. Running Health Check"

info "Waiting 30s for function app to start..."
sleep 30

HEALTH_URL="${FUNC_URL}/health"
info "Checking $HEALTH_URL"

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" --max-time 30 || echo "000")

if [[ "$HTTP_STATUS" == "200" ]]; then
  success "Health check PASSED (HTTP 200)"
elif [[ "$HTTP_STATUS" == "000" ]]; then
  warn "Health check timed out - function may still be starting"
else
  warn "Health check returned HTTP $HTTP_STATUS"
fi

# Enable logging
az webapp log config \
  --name "$FUNC_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --application-logging filesystem \
  --level information &>/dev/null || true

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}       Deployment Complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo -e "${BOLD}Function App URL:${NC} $FUNC_URL"
echo ""
echo -e "${BOLD}Test Commands:${NC}"
echo "  curl ${FUNC_URL}/health"
echo "  curl '${FUNC_URL}/recommendations/12345'"
echo ""
echo -e "${BOLD}View Logs:${NC}"
echo "  ./script/logs.sh"
echo ""
echo "=========================================="
