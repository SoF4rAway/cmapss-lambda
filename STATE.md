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
*   **Last Shell Command:** `python -m src.ingestion.generate_drift_data`
*   **Last Completed Task:** Implemented `src/ingestion/generate_drift_data.py` to programmatically inject Gaussian noise and linear scalar drift into C-MAPSS telemetry data for Lambda architecture validation. Generated `data/CMAPSSData/test_FD001_drifted.txt`.
*   **Current Active Task:** Monitoring the integration of drifted data into the Kafka ingestion pipeline.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Validate the Spark Speed Layer's resilience against the newly generated data drift.
