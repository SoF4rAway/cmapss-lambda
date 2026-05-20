---
scope: PROJECT_STATE
priority: 2 (Dynamic Context)
---

# Workspace State Tracker
*The AI must read this file before generating any implementation plan to understand its current operational context.*

## 1. Operational Context
* **Host OS:** Ubuntu WSL2 (Windows 11 Host)
* **Path Style:** POSIX (`~/cmapss-predictive-maintenance`)
* **Active Environment Manager:** Docker (Infrastructure) / Conda (Local PyTorch Training)
* **Target Environment Name:** `pytorch-gpu` (for local model authoring)

## 2. Network Mapping
* **Kafka Broker:** `localhost:9092` (via Docker WSL bridge)
* **Spark Master UI:** `http://localhost:8080`
* **Elasticsearch:** `http://localhost:9200`
* **Kibana UI:** `http://localhost:5601`

## 3. Current State
*   **Last Shell Command:** `PYTHONPATH=. python evaluate.py --model-version 20260517_135031 --no-prod-latency`
*   **Last Completed Task:** Added model version selection to `evaluate.py`/`evaluator.py`, integrated Option A fallback (local synthetic baseline), fixed a potential data leakage issue in the evaluator, and updated `speed_layer.py` to tag predictions with their generating `model_version` in Elasticsearch.
*   **Current Active Task:** Complete execution and walkthrough presentation.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Share results with the user.
