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
*   **Last Shell Command:** `conda run -n pytorch-gpu python tune_model.py`
*   **Last Completed Task:** Implemented `model.py` (1D-CNN) and `tune_model.py` (Optuna NAS) with ONNX export support.
*   **Current Active Task:** Finalized model architecture and tuning pipeline.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Integrate ONNX model into Spark Speed Layer (Streaming).
