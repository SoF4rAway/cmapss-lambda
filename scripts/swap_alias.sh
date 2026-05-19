#!/usr/bin/env bash
# scripts/swap_alias.sh
# Atomically redirects the Elasticsearch alias from v1 to v2.

ES_HOST=${1:-"localhost:9200"}

echo "Swapping alias cmapss_predictions_current from v1 to v2..."
curl -s -X POST "http://${ES_HOST}/_aliases" \
  -H 'Content-Type: application/json' \
  -d '{
    "actions": [
      {
        "remove": {
          "index": "cmapss_predictions_v1",
          "alias": "cmapss_predictions_current"
        }
      },
      {
        "add": {
          "index": "cmapss_predictions_v2",
          "alias": "cmapss_predictions_current"
        }
      }
    ]
  }'

echo -e "\nAlias swap completed."
