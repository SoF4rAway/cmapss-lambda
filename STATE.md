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
*   **Last Shell Command:** `python tune.py` (Completed 300 trials, fixed unpacking error)
*   **Last Completed Task:** Fixed `ValueError` in `tune.py` by aligning return value unpacking with the updated `run_tuning_pipeline()` (7 values).
*   **Current Active Task:** Finalizing model artifacts and verifying Pareto front visualization.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Integrate the versioned `models/{timestamp}/` artifact bundle (ONNX + Scaler + Schema) with the downstream Spark Streaming Speed Layer inference logic.
