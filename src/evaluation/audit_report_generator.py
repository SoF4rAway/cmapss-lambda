"""
src/evaluation/audit_report_generator.py
========================================
Consolidated validation coordinator that orchestrates the entire verification suite:
1. Baseline Model (V1) Evaluation (RMSE, NASA Score, Synthetic/ES Latency)
2. Retrained Model (V2) Evaluation (RMSE, NASA Score, Synthetic/ES Latency)
3. End-to-End Latency Benchmark (Canary processing duration)
4. Speed Layer Ingestion Throughput Benchmark (EPS under load)
5. Infrastructure Fault Tolerance Integration Test (SIGKILL Spark worker recovery & idempotency)

Generates a unified validation audit report saved under results/audit_{timestamp}/audit_report.md.

Usage:
    PYTHONPATH=. python src/evaluation/audit_report_generator.py --total-records 5000
"""

import os
import time
import argparse
import logging
import socket
from elasticsearch import Elasticsearch  # type: ignore[import]
from kafka import KafkaProducer  # type: ignore[import]

from src.core.config import RESULTS_DIR, set_seed
from src.evaluation.evaluator import list_model_versions, run_evaluation_pipeline
from benchmarks.throughput_test import run_throughput_benchmark
from benchmarks.e2e_latency_test import run_e2e_latency_benchmark
from tests.fault_tolerance_test import run_fault_tolerance_audit

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Audit_Report_Generator")

def preflight_checks(kafka_broker: str, es_host: str, skip_ft: bool) -> bool:
    """Verifies infrastructure components are accessible before launching the suite."""
    logger.info("Executing pre-flight checks on infrastructure...")
    
    # 1. Check Elasticsearch
    try:
        es = Elasticsearch(es_host)
        if not es.ping():
            logger.error(f"Elasticsearch at {es_host} is not responding to pings.")
            return False
        logger.info("Elasticsearch connection: OK")
    except Exception as e:
        logger.error(f"Elasticsearch pre-flight check failed: {e}")
        return False

    # 2. Check Kafka
    try:
        # Resolve broker address
        host, port_str = kafka_broker.split(":")
        port = int(port_str)
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        logger.info("Kafka connection: OK")
    except Exception as e:
        logger.error(f"Kafka broker at {kafka_broker} is unreachable: {e}")
        return False

    # 3. Check Docker (if running fault tolerance)
    if not skip_ft:
        try:
            import docker
            client = docker.from_env()
            client.ping()
            logger.info("Docker daemon connection: OK")
        except Exception as e:
            logger.error(f"Docker daemon connection failed (required for fault tolerance test): {e}")
            return False

    return True

def run_audit(args) -> None:
    # Set seed
    set_seed(42)
    
    # Pre-flight check
    if not preflight_checks(args.kafka_broker, args.es_host, args.skip_fault_tolerance):
        logger.error("Pre-flight system checks failed. Aborting audit report generation.")
        exit(1)

    # 1. Version Detection
    versions = list_model_versions()
    logger.info(f"Available model versions in directory: {versions}")

    v1_version = args.v1_version
    v2_version = args.v2_version

    if not v1_version:
        if len(versions) >= 2:
            v1_version = versions[-2]
            logger.info(f"Auto-detected Baseline model (V1): {v1_version}")
        elif len(versions) == 1:
            v1_version = versions[-1]
            logger.warning(f"Only one model version found. Using {v1_version} for Baseline (V1)")
        else:
            logger.error("No model versions found to evaluate.")
            exit(1)

    if not v2_version:
        if len(versions) >= 1:
            v2_version = versions[-1]
            logger.info(f"Auto-detected Retrained model (V2): {v2_version}")
        else:
            logger.error("No model versions found to evaluate.")
            exit(1)

    # Output directory
    audit_timestamp = time.strftime("%Y%m%d_%H%M%S")
    audit_results_dir = os.path.join(RESULTS_DIR, f"audit_{audit_timestamp}")
    os.makedirs(audit_results_dir, exist_ok=True)

    # 2. Evaluate Baseline Model V1
    logger.info(f"Starting Baseline Model (V1) Evaluation [Version: {v1_version}]...")
    v1_metrics = None
    try:
        v1_metrics = run_evaluation_pipeline(
            model_timestamp=v1_version, 
            fetch_prod_latency=True, 
            es_index=args.es_index_v1
        )
    except Exception as e:
        logger.exception(f"Failed to evaluate Baseline Model (V1): {e}")

    # 3. Evaluate Retrained Model V2
    logger.info(f"Starting Retrained Model (V2) Evaluation [Version: {v2_version}]...")
    v2_metrics = None
    try:
        v2_metrics = run_evaluation_pipeline(
            model_timestamp=v2_version, 
            fetch_prod_latency=True, 
            es_index=args.es_index_v2
        )
    except Exception as e:
        logger.exception(f"Failed to evaluate Retrained Model (V2): {e}")

    # 4. Ingestion Throughput Test
    throughput_metrics = None
    if not args.skip_throughput:
        logger.info("Starting Speed Layer Ingestion Throughput Benchmark...")
        try:
            # Running throughput benchmark against active retrained index v2
            throughput_metrics = run_throughput_benchmark(
                es_index=args.es_index_v2,
                kafka_broker=args.kafka_broker,
                es_host=args.es_host,
                total_records=args.total_records
            )
        except Exception as e:
            logger.error(f"Throughput test execution encountered error: {e}")
            throughput_metrics = {"eps": 0.0, "processing_time": 0.0, "total_records": args.total_records, "error": str(e)}

    # 5. E2E Latency Test
    e2e_metrics = None
    if not args.skip_e2e:
        logger.info("Starting E2E Latency Benchmark...")
        try:
            # Running E2E test against active retrained index v2
            e2e_metrics = run_e2e_latency_benchmark(
                es_index=args.es_index_v2,
                kafka_broker=args.kafka_broker,
                es_host=args.es_host
            )
        except Exception as e:
            logger.error(f"E2E Latency test execution encountered error: {e}")
            e2e_metrics = {"e2e_latency": -1.0, "attempts": 0, "error": str(e)}

    # 6. Fault Tolerance Test
    ft_metrics = None
    if not args.skip_fault_tolerance:
        logger.info("Starting Fault Tolerance Integration Test...")
        try:
            ft_metrics = run_fault_tolerance_audit(
                es_index=args.es_index_v2,
                kafka_broker=args.kafka_broker,
                es_host=args.es_host,
                worker_name=args.spark_worker_name,
                timeout=args.timeout
            )
        except Exception as e:
            logger.error(f"Fault Tolerance test execution encountered error: {e}")
            ft_metrics = {"test_passed": False, "recovery_latency": -1.0, "consistency_result": f"FAIL (Error: {e})"}

    # 7. Write consolidated report
    report_path = os.path.join(audit_results_dir, "audit_report.md")
    
    # Calculate Deltas
    v1_rmse = v1_metrics["rmse"] if v1_metrics else float('nan')
    v2_rmse = v2_metrics["rmse"] if v2_metrics else float('nan')
    rmse_diff = v2_rmse - v1_rmse
    
    v1_nasa = v1_metrics["nasa_score"] if v1_metrics else float('nan')
    v2_nasa = v2_metrics["nasa_score"] if v2_metrics else float('nan')
    nasa_diff = v2_nasa - v1_nasa
    nasa_percent_diff = ((nasa_diff / v1_nasa) * 100) if v1_nasa != 0 else float('nan')

    # Circuit breaker check: Retrained NASA score must be within 5% of the baseline score.
    circuit_breaker_threshold = v1_nasa * 1.05
    circuit_breaker_passed = v2_nasa <= circuit_breaker_threshold if (v1_nasa and v2_nasa) else False
    
    report_content = f"""# Lambda Architecture Verification & Performance Audit
Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
Audit Run Directory: `results/audit_{audit_timestamp}/`

---

## 1. Executive Summary & Verification Decisions

| Metric / Check | Baseline (V1) | Retrained (V2) | Change / Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Model Version (Timestamp)** | `{v1_version}` | `{v2_version}` | - | - |
| **Model RMSE** | `{v1_rmse:.4f}` | `{v2_rmse:.4f}` | `{rmse_diff:+.4f}` | {"PASS" if v2_rmse <= v1_rmse else "WARNING (Degraded)"} |
| **NASA Asymmetric Score** | `{v1_nasa:.2f}` | `{v2_nasa:.2f}` | `{nasa_diff:+.2f} ({nasa_percent_diff:+.2f}%)` | {"PASS" if circuit_breaker_passed else "FAIL (Retrain Blocked)"} |
| **5% Retrain Circuit Breaker** | - | - | Limit: `<= {circuit_breaker_threshold:.2f}` | **{"PASSED" if circuit_breaker_passed else "FAILED"}** |

> [!NOTE]
> The performance circuit breaker threshold enforces that the retrained model (V2) NASA score must not exceed the baseline model (V1) score by more than **5.0%** (Limit: `{circuit_breaker_threshold:.2f}`).

---

## 2. Speed Layer Real-Time Benchmarks

### Ingestion Throughput
*   **Total Stress Records Injected**: `{throughput_metrics.get("total_records", "N/A") if throughput_metrics else "N/A"}`
*   **Total Processing Duration**: `{f'{throughput_metrics.get("processing_time", 0.0):.2f} seconds' if throughput_metrics and throughput_metrics.get("processing_time") else "N/A"}`
*   **Throughput**: **`{f'{throughput_metrics.get("eps", 0.0):.2f} EPS' if throughput_metrics and throughput_metrics.get("eps") else "N/A"}`**
*   **Status**: `{"PASS" if throughput_metrics and not throughput_metrics.get("error") else "FAILED/SKIPPED"}`

### End-to-End Latency
*   **E2E Visibility Latency**: **`{f'{e2e_metrics.get("e2e_latency", 0.0):.4f} seconds' if e2e_metrics and e2e_metrics.get("e2e_latency") and e2e_metrics.get("e2e_latency") > 0 else "N/A"}`**
*   **ES Polling Attempts**: `{e2e_metrics.get("attempts", "N/A") if e2e_metrics else "N/A"}`
*   **Status**: `{"PASS" if e2e_metrics and e2e_metrics.get("e2e_latency", -1) > 0 else "FAILED/SKIPPED"}`

---

## 3. Infrastructure & Fault Tolerance Audit

*   **Target Worker Container**: `{args.spark_worker_name}`
*   **SIGKILL Injected**: `{"Yes" if ft_metrics else "No"}`
*   **Recovery Latency**: `{f'{ft_metrics.get("recovery_latency", 0.0):.2f} seconds' if ft_metrics and ft_metrics.get("recovery_latency", -1) > 0 else "N/A"}`
*   **Data Consistency**: `{ft_metrics.get("consistency_result", "N/A") if ft_metrics else "N/A"}`
*   **Fault Tolerance Result**: **`{"PASS" if ft_metrics and ft_metrics.get("test_passed") else "FAIL/SKIPPED"}`**

---

## 4. Comprehensive Metric Comparison

### Model Artifact Sizes
*   **Baseline V1 Model Size**: `{f'{v1_metrics.get("model_size_kb", 0.0):.2f} KB' if v1_metrics else "N/A"}`
*   **Retrained V2 Model Size**: `{f'{v2_metrics.get("model_size_kb", 0.0):.2f} KB' if v2_metrics else "N/A"}`

### Error Statistics (Residual Analysis)
*   **Baseline V1 Bias (Mean Error)**: `{v1_metrics.get("error_mean", "N/A") if v1_metrics else "N/A"}`
*   **Retrained V2 Bias (Mean Error)**: `{v2_metrics.get("error_mean", "N/A") if v2_metrics else "N/A"}`
*   **Baseline V1 Error Std**: `{v1_metrics.get("error_std", "N/A") if v1_metrics else "N/A"}`
*   **Retrained V2 Error Std**: `{v2_metrics.get("error_std", "N/A") if v2_metrics else "N/A"}`

---

## 5. Visualizations & Paths
*   **V1 Performance Report**: `{v1_metrics.get("report_path", "N/A") if v1_metrics else "N/A"}`
*   **V1 Performance Plot**: `{v1_metrics.get("plot_path", "N/A") if v1_metrics else "N/A"}`
*   **V2 Performance Report**: `{v2_metrics.get("report_path", "N/A") if v2_metrics else "N/A"}`
*   **V2 Performance Plot**: `{v2_metrics.get("plot_path", "N/A") if v2_metrics else "N/A"}`
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(report_content)
    logger.info(f"Consolidated Validation Audit Report saved to: {report_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="CMAPSS Lambda Architecture Audit Coordinator")
    parser.add_argument("--v1-version", default=None, help="Baseline model version timestamp (e.g. 20260517_145257).")
    parser.add_argument("--v2-version", default=None, help="Retrained model version timestamp. Defaults to latest model.")
    parser.add_argument("--es-index-v1", default="cmapss_predictions_v1", help="Elasticsearch index for V1 predictions.")
    parser.add_argument("--es-index-v2", default="cmapss_predictions_v2", help="Elasticsearch index for V2 predictions.")
    parser.add_argument("--kafka-broker", default="localhost:9092", help="Kafka broker address.")
    parser.add_argument("--es-host", default="http://localhost:9200", help="Elasticsearch host address.")
    parser.add_argument("--total-records", type=int, default=50000, help="Total stress test records to flood.")
    parser.add_argument("--timeout", type=int, default=120, help="Fault tolerance recovery timeout.")
    parser.add_argument("--spark-worker-name", default="spark-worker", help="Spark worker container name to kill.")
    
    # Flags to skip individual long-running/destructive tests
    parser.add_argument("--skip-throughput", action="store_true", help="Skip the stress throughput benchmark.")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip the E2E visibility latency benchmark.")
    parser.add_argument("--skip-fault-tolerance", action="store_true", help="Skip the Spark worker fault tolerance test.")
    
    args = parser.parse_args()
    
    run_audit(args)

if __name__ == "__main__":
    main()
