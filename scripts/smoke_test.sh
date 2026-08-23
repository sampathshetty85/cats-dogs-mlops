#!/bin/bash
set -e

BASE_URL="${1:-http://localhost:8000}"

echo "Running smoke tests against $BASE_URL ..."

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
[ "$STATUS" = "200" ] || { echo "FAIL: /health returned $STATUS"; exit 1; }
echo "  /health OK (200)"

RESPONSE=$(curl -s -F "file=@tests/fixtures/sample_cat.jpg" "$BASE_URL/predict")
echo "$RESPONSE" | grep -q '"label"' || { echo "FAIL: /predict response missing label: $RESPONSE"; exit 1; }
echo "  /predict OK — $RESPONSE"

echo "Smoke tests passed."
