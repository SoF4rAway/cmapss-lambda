import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import logging
from typing import Tuple
from src.core.config import set_seed, DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.data.preprocess import CMAPSSPreprocessor
from src.data.loaders import CMAPSSTestDataset
from src.training.utils import nasa_asymmetric_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_Evaluation")

def benchmark_inference(onnx_path: str, input_shape: tuple, n_runs: int = 1000) -> Tuple[float, float]:
    """Benchmarks inference latency and throughput."""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    start_time = time.perf_counter()
    for _ in range(n_runs):
        _ = session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) / n_runs) * 1000  # ms
    throughput = n_runs / (end_time - start_time)  # samples/sec
    return avg_latency, throughput

def get_latest_model(models_dir=MODELS_DIR):
    if not os.path.exists(models_dir):
        return None
    models = sorted([f for f in os.listdir(models_dir) if f.endswith(".onnx")])
    if not models:
        return None
    return os.path.join(models_dir, models[-1])

def run_evaluation_pipeline():
    # 1. Setup paths
    CMAPSS_DATA_DIR = os.path.join(DATA_DIR, "CMAPSSData")
    test_path = os.path.join(CMAPSS_DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(CMAPSS_DATA_DIR, "RUL_FD001.txt")
    train_path = os.path.join(CMAPSS_DATA_DIR, "train_FD001.txt")
    
    onnx_model_path = get_latest_model()
    if not onnx_model_path or not os.path.exists(onnx_model_path):
        logger.error("No ONNX models found.")
        return
    
    # Extract timestamp for versioned results
    model_filename = os.path.basename(onnx_model_path)
    timestamp = model_filename.split("_rul_model")[0] if "_rul_model" in model_filename else "latest"
    run_results_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(run_results_dir, exist_ok=True)
    
    logger.info(f"Evaluating model: {onnx_model_path}")

    # 2. Data Loading
    preprocessor = CMAPSSPreprocessor(max_rul=125)
    raw_train = preprocessor.load_data(train_path)
    raw_train = preprocessor.add_piecewise_rul(raw_train)
    train_set, _ = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2, seed=42)
    _ = preprocessor.fit_transform(train_set)
    
    test_labeled = preprocessor.load_and_label_test_data(test_path, rul_path)
    test_scaled = preprocessor.transform(test_labeled)
    test_ds = CMAPSSTestDataset(test_scaled, sequence_length=30)
    x_test = test_ds.features.numpy()
    y_true = test_ds.labels.numpy().flatten()
    
    # 3. Inference
    session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    y_pred = session.run(None, {input_name: x_test})[0].flatten()
    
    # 4. Metrics
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    nasa_score = nasa_asymmetric_score(y_pred, y_true)
    errors = y_pred - y_true
    
    input_shape = (1, 30, x_test.shape[2])
    avg_latency, throughput = benchmark_inference(onnx_model_path, input_shape)
    model_size_kb = os.path.getsize(onnx_model_path) / 1024
    
    # 5. Visualizations
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=ax1, color='#2c3e50')
    ax1.plot([0, 140], [0, 140], '--', color='#e74c3c', linewidth=2)
    sns.histplot(errors, kde=True, ax=ax2, color='#3498db', bins=20)
    plt.tight_layout()
    
    plot_path = os.path.join(run_results_dir, "academic_performance_results.png")
    plt.savefig(plot_path, dpi=300)
    
    # 6. Report Generation
    report = f"""# Academic Performance Report: {timestamp}
- Model: {model_filename}
- RMSE: {rmse:.4f}
- NASA Score: {nasa_score:.2f}
- Latency: {avg_latency:.4f} ms
- Model Size: {model_size_kb:.2f} KB
"""
    report_path = os.path.join(run_results_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(report)
    logger.info(f"Results saved to {run_results_dir}")

if __name__ == "__main__":
    set_seed(42)
    run_evaluation_pipeline()
