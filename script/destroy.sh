#!/usr/bin/env bash
set -euo pipefail

STACK="${1:-dev}"
ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PULUMI_PROJECT="$(awk '/^name:/ {print $2; exit}' "$ROOT_DIR/infra/Pulumi.yaml")"
RESOURCE_GROUP="$PULUMI_PROJECT"
SUBSCRIPTION_ID="$(az account show --query id -o tsv 2>/dev/null || echo "")"

echo "=========================================="
echo "Destroying Infrastructure"
echo "=========================================="
echo "Stack: $STACK"
echo "Project: $PULUMI_PROJECT"
echo ""

# Clean up Application Insights Smart Detection if it exists
if [[ -n "$SUBSCRIPTION_ID" ]]; then
  SMART_DETECTION_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/microsoft.insights/actiongroups/Application Insights Smart Detection"
  if az resource show --ids "$SMART_DETECTION_ID" >/dev/null 2>&1; then
    echo "[destroy] Deleting Application Insights Smart Detection action group"
    az resource delete --ids "$SMART_DETECTION_ID" || true
  fi
fi

cd "$ROOT_DIR/infra"
if pulumi stack select "$STACK" >/dev/null 2>&1; then
  echo "[destroy] Running pulumi destroy on stack ${STACK}"
  pulumi destroy --stack "$STACK" --yes
  pulumi stack rm "$STACK" --yes || true
else
  echo "[destroy] Pulumi stack ${STACK} not found, skipping"
fi

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
