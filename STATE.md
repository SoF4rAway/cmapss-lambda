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
*   **Last Shell Command:** `python -m py_compile src/streaming/speed_layer.py`
*   **Last Completed Task:** Implemented a Multi-Index Strategy in `speed_layer.py` by supporting the `--es-index` argument and using `functools.partial` to dynamically pass the target index to `foreachBatch`. Created Elasticsearch alias initialization (`init_alias.sh`) and swap (`swap_alias.sh`) scripts to atomically map Kibana Dashboards via `cmapss_predictions_current`.
*   **Current Active Task:** Pipeline validation prep and documenting commands.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Create walkthrough.md and present commands to the user.
