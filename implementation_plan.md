# Alignment Scan between ARCHITECTURE.md and Codebase

This document details the discrepancies found between the stated architecture in `ARCHITECTURE.md` and the actual implementation within the codebase. Per the request, no files have been modified yet. This plan serves to document all misalignments and propose corrections.

## User Review Required

> [!IMPORTANT]
> Please review the listed discrepancies below. Let me know if you would like me to proceed with updating `ARCHITECTURE.md` to reflect the codebase's true state, or if any of the codebase implementations should be modified to match the existing documentation. 

## Proposed Changes

We will modify `ARCHITECTURE.md` to reflect the current reality of the codebase. As per your instruction, we will update the documentation to match the stricter 5% threshold implemented in the circuit breaker, and fix all other misalignments.

### Documentation

#### [MODIFY] [ARCHITECTURE.md](file:///mnt/d/Repositories/personal-projects/ai-ml-projects/cmapss-lambda/ARCHITECTURE.md)
We will make the following specific edits to this file:

1. **Section 3 (Temporal Transformation)**:
   - Replace the file reference `src/models/rul_cnn.py` with the correct path `src/models/architecture.py`.

2. **Section 4 (Lambda Pipeline Phases - Speed Layer)**:
   - Remove the phrase "and 100MB block sizes" from the HDFS Cold Path description since it's not configured in the code.
   - Remove the explicit line number reference ("See lines 329+ in src/streaming/speed_layer.py") for state initialization to prevent future staleness, replacing it with a reference to the `make_predict_rul_udf` closure.

3. **Section 4 (Lambda Pipeline Phases - Batch Layer)**:
   - Update the retraining trigger condition from "10% worse than baseline (notBias > 10.0)" to "5% worse than baseline" to match the strict 5% threshold in the code (`max_allowed_score = baseline_score * 1.05`).
   - Remove the reference to the non-existent `compare_nasa_scores()` function and specify that the check occurs inline within `run_batch_retrain()`.

4. **Section 8 (Implementation Verification Notes)**:
   - Update the `Sliding Window Transpose` verification file from `src/models/rul_cnn.py` to `src/models/architecture.py`.
   - Update the `Sliding Window Transpose` key function from `RUL1DCNN.forward()` to `RUL_1D_CNN.forward()`.
   - Update the `Retraining Trigger` key function from `compare_nasa_scores()` to `run_batch_retrain()`.

---

## Verification Plan

### Manual Verification
- Review the proposed updates to `ARCHITECTURE.md` to ensure they align with the project's long-term design goals.
- Approve the plan to proceed with updating the documentation (or adjusting the code if preferred).
