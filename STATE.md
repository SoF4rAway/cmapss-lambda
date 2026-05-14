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
*   **Last Shell Command:** `docker exec -it spark-master spark-submit --packages ... src/streaming/speed_layer.py ...`
*   **Last Completed Task:** Fully synchronized [ARCHITECTURE.md](file:///mnt/d/Repositories/personal-projects/ai-ml-projects/cmapss-lambda/ARCHITECTURE.md) with the current codebase, documenting the Zero Data Loss sliding window logic, Tiered Sink threading strategy, and `shuffle.partitions` performance tuning.
*   **Current Active Task:** System is fully optimized, documented, and operating in production mode.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Monitor for long-term drift or potential scale-up of the worker nodes.
