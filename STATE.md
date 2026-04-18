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
*   **Last Shell Command:** `conda run -n pytorch-gpu python preprocess_cmapss.py`
*   **Last Completed Task:** Implemented PyTorch `CMAPSSTrainDataset`, `CMAPSSTestDataset`, and `get_dataloaders` wrapper in `data_loaders.py`.
*   **Current Active Task:** Verifying DataLoader logic and pre-padding functionality.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Implement the PyTorch model architecture (LSTM/1D-CNN) and training loop.
