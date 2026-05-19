#!/usr/bin/env bash
# scripts/init_alias.sh
# Initializes the Elasticsearch v1 index and points the Kibana-monitored alias to it.

ES_HOST=${1:-"localhost:9200"}

echo "Creating index cmapss_predictions_v1..."
curl -s -X PUT "http://${ES_HOST}/cmapss_predictions_v1" \
  -H 'Content-Type: application/json' \
  -d '{}'

echo -e "\nPointing alias cmapss_predictions_current -> cmapss_predictions_v1..."
curl -s -X POST "http://${ES_HOST}/_aliases" \
  -H 'Content-Type: application/json' \
  -d '{
    "actions": [
      {
        "add": {
          "index": "cmapss_predictions_v1",
          "alias": "cmapss_predictions_current"
        }
      }
    ]
  }'

echo -e "\nAlias initialization completed."
