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
*   **Last Shell Command:** `conda run -n pytorch-gpu env PYTHONPATH=. python -m src.batch.batch_retrain`
*   **Last Completed Task:** Refactored `batch_retrain.py` to implement source-based sample weighting (1.0 for baseline, 0.3 for HDFS data) and locked the `StandardScaler` to baseline model statistics by loading it in Transform Mode. Adapted the PyTorch dataset loaders and `asymmetric_loss` to support element-wise sample weights while preserving backward compatibility with `tuner.py`. Verified functionality via test scripts.
*   **Current Active Task:** None. Completed continuous learning training stability refactoring.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Await further tasks from the user.
