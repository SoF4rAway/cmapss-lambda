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
*   **CUDA Available:** [Agent to verify: e.g., True — RTX 5060 Ti detected]
*   **PyTorch Version:** [Agent to fill: e.g., 2.x.x+cu121]

## 2. Project Overview & Network Mapping
* **Kafka Broker:** `localhost:9092` (via Docker WSL bridge)
* **Spark Master UI:** `http://localhost:8080`
* **Elasticsearch:** `http://localhost:9200`
* **Kibana UI:** `http://localhost:5601`
*   **Project Name:** [e.g., RetinaNet Custom Detector]
*   **Task Type:** [e.g., Object Detection / Image Classification / NLP / Tabular / etc.]
*   **Dataset:** [e.g., COCO subset — 5k images, stored at data/raw/coco_subset/]
*   **Model Architecture:** [e.g., ResNet-50 backbone + FPN]
*   **Training Status:** [e.g., Not started / Epoch 12 of 50 / Complete]

## 3. Current State
*   **Last Shell Command:** `PYTHONPATH=. python evaluate.py --model-version 20260517_135031 --no-prod-latency`
*   **Last Completed Task:** Added model version selection to `evaluate.py`/`evaluator.py`, integrated Option A fallback (local synthetic baseline), fixed a potential data leakage issue in the evaluator, and updated `speed_layer.py` to tag predictions with their generating `model_version` in Elasticsearch.
*   **Current Active Task:** Complete execution and walkthrough presentation.
*   **Known Blockers/Issues:** None.

## 4. Key File Locations
*   **Config File:** [e.g., configs/train_config.yaml]
*   **Main Training Script:** [e.g., src/training/train.py]
*   **Latest Checkpoint:** [e.g., checkpoints/epoch_12_val_loss_0.342.pt]
*   **Latest Output/Log:** [e.g., outputs/run_20250520/]

## 5. Continuity Note
*   **Next Planned Step:** [Agent to update: e.g., Run evaluation on val split and log mAP]
*   **Decisions Made:** [Agent to update: e.g., Chose AdamW over SGD due to faster convergence on this dataset size]
