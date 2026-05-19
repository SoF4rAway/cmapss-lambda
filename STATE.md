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
*   **Last Shell Command:** `python -m src.batch.batch_retrain`
*   **Last Completed Task:** Implemented the Batch Layer Retraining Pipeline (`src/batch/batch_retrain.py`) that loads historical streaming telemetry from HDFS, combines it with baseline data, refits scaler parameters, loads the correct model architecture using dynamically loaded hyperparameters, fine-tunes the PyTorch model, and exports dual artifacts.
*   **Current Active Task:** Verifying the execution and compatibility of the batch retraining script.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Run a full test of the batch retrain pipeline and verify the versioned output directory is created successfully.
