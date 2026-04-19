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
*   **Last Shell Command:** N/A (code authoring session)
*   **Last Completed Task:** Implemented strict Artifact Versioning in `CMAPSSPreprocessor` and synchronized `tuner.py` and `loaders.py` to use a deterministic feature schema.
    *   `src/data/preprocess.py`: Dual Fit/Transform modes via `artifact_dir`. New `save_artifacts()` method bundles `scaler.joblib` + `feature_schema.json`. Hardcoded `PROJECT_ROOT` schema export removed.
    *   `src/data/loaders.py`: Removed hardcoded feature exclusion list comprehensions. `feature_cols: List[str]` is now a mandatory argument to all Dataset classes and `get_dataloaders()`.
    *   `src/training/tuner.py`: `Objective` receives `feature_cols` explicitly. `run_tuning_pipeline()` returns `preprocessor` + `feature_cols`. `finalize_model()` creates a versioned `models/{timestamp}/` directory and calls `preprocessor.save_artifacts()`.
    *   `requirements.txt`: Created with `joblib` explicitly listed for environment reproducibility.
*   **Current Active Task:** None.
*   **Known Blockers/Issues:** None.
*   **Next Planned Step:** Integrate the versioned `models/{timestamp}/` artifact bundle with the downstream Spark Streaming Speed Layer inference logic.
