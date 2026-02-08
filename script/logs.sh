#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PULUMI_PROJECT="$(awk '/^name:/ {print $2; exit}' "$ROOT_DIR/infra/Pulumi.yaml")"
RESOURCE_GROUP="$PULUMI_PROJECT"
FUNC_NAME="${PULUMI_PROJECT}-func"
TIMEOUT="${1:-60}"

echo "Streaming logs for $FUNC_NAME (timeout: ${TIMEOUT}s)..."
echo "Press Ctrl+C to stop"
echo ""

az webapp log tail --name "$FUNC_NAME" --resource-group "$RESOURCE_GROUP" --timeout "$TIMEOUT"
