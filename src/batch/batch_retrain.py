"""
Batch Layer Retraining Pipeline.

Performs automated model retraining (warm-start) on a combination of baseline
clean data and drifted historical telemetry collected from the HDFS cold path.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    SparkSession = None
    HAS_PYSPARK = False

from src.core.config import set_seed, DATA_DIR, MODELS_DIR
from src.models.architecture import RUL_1D_CNN, export_to_onnx
from src.data.loaders import get_dataloaders
from src.data.preprocess import CMAPSSPreprocessor
from src.training.utils import asymmetric_loss

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BatchRetrain")


def get_latest_model_dir(models_dir: str) -> str:
    """
    Dynamically discover the most recent baseline model artifact directory.

    Args:
        models_dir: Base models directory containing versioned timestamp folders.

    Returns:
        The absolute path of the latest baseline model directory.
    """
    subdirs = [
        os.path.join(models_dir, d)
        for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]
    valid_dirs = []
    for d in subdirs:
        # Ignore target output directory for drift correction to prevent self-referencing
        if "v2_drift_corrected" in d:
            continue
        if os.path.exists(os.path.join(d, "model.pth")):
            valid_dirs.append(d)

    if not valid_dirs:
        raise FileNotFoundError(
            f"No baseline model directory containing 'model.pth' found in {models_dir}"
        )

    # Timestamps are named in chronological string order (e.g. YYYYMMDD_HHMMSS)
    valid_dirs.sort()
    latest_dir = valid_dirs[-1]
    logger.info(f"Identified latest baseline model directory: {latest_dir}")
    return latest_dir


def load_hdfs_telemetry(spark: Optional[SparkSession], hdfs_path: str) -> Optional[pd.DataFrame]:
    """
    Load historical telemetry data from HDFS or local CSV fallback and align schema.

    Drops Speed Layer prediction tracking columns and renames telemetry time index.

    Args:
        spark: Active Spark session (optional).
        hdfs_path: Path to the HDFS Parquet dataset.

    Returns:
        Aligned pandas DataFrame or None if loading fails.
    """
    local_fallback = os.path.join(DATA_DIR, "CMAPSSData", "hdfs_telemetry.csv")
    
    # If Spark is not available, fall back to the pre-exported CSV
    if spark is None:
        if os.path.exists(local_fallback):
            logger.info(f"PySpark not available. Loading pre-exported HDFS telemetry from fallback: {local_fallback}")
            try:
                df_pd = pd.read_csv(local_fallback)
                logger.info(f"Successfully loaded fallback telemetry. Shape: {df_pd.shape}")
                return df_pd
            except Exception as e:
                logger.error(f"Failed to load fallback telemetry: {e}")
                return None
        else:
            logger.warning(f"PySpark not available and local fallback {local_fallback} does not exist.")
            return None

    try:
        logger.info(f"Reading HDFS cold path data from: {hdfs_path}")
        hdfs_df = spark.read.parquet(hdfs_path)
        
        # Identify and drop streaming-specific output tracking columns
        tracking_cols = ["predicted_rul", "is_critical", "doc_id"]
        drop_cols = [c for c in tracking_cols if c in hdfs_df.columns]
        if drop_cols:
            hdfs_df = hdfs_df.drop(*drop_cols)
            logger.info(f"Dropped tracking columns: {drop_cols}")

        # Convert to Pandas DataFrame
        df_pd = hdfs_df.toPandas()
        
        # Rename streaming time cycle index to match train_FD001.txt schema
        if "time_cycles" in df_pd.columns:
            df_pd = df_pd.rename(columns={"time_cycles": "cycle"})
            
        logger.info(f"Successfully loaded and aligned HDFS telemetry. Shape: {df_pd.shape}")
        return df_pd
    except Exception as e:
        logger.error(f"Failed to load HDFS data: {e}")
        # Try local CSV fallback before giving up
        if os.path.exists(local_fallback):
            logger.info(f"Spark load failed. Falling back to local telemetry CSV: {local_fallback}")
            try:
                df_pd = pd.read_csv(local_fallback)
                return df_pd
            except Exception as read_err:
                logger.error(f"Failed to read local fallback CSV: {read_err}")
        return None


def run_batch_retrain(epochs: int = 5, lr: float = 1e-4, weight_decay: float = 1e-2) -> str:
    """
    Perform the batch retraining pipeline.

    Integrates HDFS streaming telemetry with baseline data, refits scaler,
    warm-starts model weights, fine-tunes, and exports new artifacts.
    """
    set_seed(42)

    # 1. Initialize optimized SparkSession if PySpark is available
    spark = None
    if HAS_PYSPARK:
        try:
            spark = SparkSession.builder \
                .appName("CMAPSS-Batch-Retraining") \
                .config("spark.sql.shuffle.partitions", "2") \
                .getOrCreate()
            logger.info("Successfully initialized SparkSession.")
        except Exception as e:
            logger.warning(f"Failed to initialize SparkSession: {e}. Degrading to pandas fallback.")

    # 2. Load and align datasets
    hdfs_path = "hdfs://namenode:9000/cmapss/batch_layer/predictions/"
    hdfs_pd = load_hdfs_telemetry(spark, hdfs_path)

    # Load baseline dataset
    preprocessor = CMAPSSPreprocessor(max_rul=125)
    base_train_path = os.path.join(DATA_DIR, "CMAPSSData", "train_FD001.txt")
    base_pd = preprocessor.load_data(base_train_path)

    if hdfs_pd is not None and not hdfs_pd.empty:
        # Prevent unit_id collision between baseline and streaming engines
        max_base_unit = base_pd["unit_id"].max()
        hdfs_pd["unit_id"] = hdfs_pd["unit_id"] + max_base_unit
        
        # Combine datasets
        combined_pd = pd.concat([base_pd, hdfs_pd], ignore_index=True)
        logger.info(f"Merged base and HDFS datasets. Total engines: {combined_pd['unit_id'].nunique()}")
    else:
        logger.warning("No HDFS streaming data integrated. Retraining strictly on baseline clean data.")
        combined_pd = base_pd

    # 3. Dynamic Model Directory Discovery & Hyperparameter loading
    latest_model_dir = get_latest_model_dir(MODELS_DIR)
    
    # Load feature schema to ensure exact column mapping matches the baseline network dimensions
    schema_path = os.path.join(latest_model_dir, "feature_schema.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    active_features = schema["active_features"]
    logger.info(f"Loaded schema with {len(active_features)} active features.")

    # Load hyperparameters to instantiate the exact model structure
    hparams_path = os.path.join(latest_model_dir, "hyperparameters.json")
    if os.path.exists(hparams_path):
        with open(hparams_path, "r") as f:
            best_params = json.load(f)
        logger.info(f"Loaded exact model hyperparameters from {hparams_path}")
    else:
        logger.warning(f"hyperparameters.json not found in {latest_model_dir}. Falling back to default configuration.")
        best_params = {
            "num_blocks": 2,
            "kernel_size": 7,
            "dilation": 1,
            "fc_units": 64,
            "out_channels_b0": 16,
            "out_channels_b1": 16,
            "use_bn_b0": False,
            "use_bn_b1": True,
            "dropout_b0": 0.1,
            "dropout_b1": 0.1
        }

    # 4. Feature Engineering & Vectorized Scaling
    # Apply Piecewise Linear RUL Target labeling, capping truth values at 125 cycles
    combined_pd = preprocessor.add_piecewise_rul(combined_pd)

    # Split combined engines into train/validation to prevent temporal leakage
    train_set, val_set = preprocessor.split_train_val_by_engine(combined_pd, val_ratio=0.1, seed=42)

    # Re-fit the StandardScaler on the new combined training dataset to update global statistics
    preprocessor.active_features = active_features
    preprocessor.scaler.fit(train_set[active_features])

    # Transform training and validation sets using the newly fitted scaler
    train_scaled = preprocessor.transform(train_set)
    val_scaled = preprocessor.transform(val_set)

    # Create 3D sliding window tensors (batch, channels, 30) grouped by unit_id
    train_loader, val_loader, _ = get_dataloaders(
        train_scaled,
        val_scaled,
        val_scaled,  # Dummy test loader placeholder
        feature_cols=active_features,
        sequence_length=30,
        batch_size=64,
        seed=42
    )

    # 5. Instantiate CNN-SE Architecture & Load baseline weights
    num_blocks = best_params["num_blocks"]
    out_channels_list = [best_params[f"out_channels_b{i}"] for i in range(num_blocks)]
    use_bn_list = [best_params[f"use_bn_b{i}"] for i in range(num_blocks)]
    dropout_list = [best_params[f"dropout_b{i}"] for i in range(num_blocks)]

    model = RUL_1D_CNN(
        input_channels=len(active_features),
        num_blocks=num_blocks,
        out_channels_list=out_channels_list,
        kernel_size=best_params["kernel_size"],
        dilation=best_params.get("dilation", 1),
        use_bn_list=use_bn_list,
        dropout_list=dropout_list,
        fc_units=best_params["fc_units"]
    )

    # Load baseline model state dict
    model_path = os.path.join(latest_model_dir, "model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    logger.info(f"Loaded pre-trained weights from {model_path} on {device}")

    # 6. Warm-Start Retraining Loop
    # Use conservative optimization parameters to adapt to operational drift gently
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = asymmetric_loss

    logger.info(f"Starting fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss += criterion(outputs, y).item() * x.size(0)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        
        logger.info(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss (asym): {epoch_loss:.4f} | "
            f"Val Loss (asym): {val_epoch_loss:.4f}"
        )

    # 7. Dual-Artifact Export & Deployment Preparation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODELS_DIR, "v2_drift_corrected", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # A. Save PyTorch weights
    pth_save_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), pth_save_path)
    logger.info(f"Saved warm-started weights to '{pth_save_path}'.")

    # B. Export model graph to optimized ONNX format (Opset 14)
    onnx_save_path = os.path.join(save_dir, "model.onnx")
    export_to_onnx(model.cpu(), onnx_save_path, input_shape=(1, 30, len(active_features)))
    logger.info(f"Exported fine-tuned ONNX graph to '{onnx_save_path}'.")

    # C. Save scaler.joblib and feature_schema.json
    preprocessor.save_artifacts(save_dir)

    # D. Bundle hyperparameters.json for exact downstream replication
    with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info("Bundled hyperparameters.json for version continuity.")

    logger.info(f"Batch retraining complete. Versioned artifact saved to: {save_dir}")
    
    # Shutdown Spark session if it was initialized
    if spark is not None:
        spark.stop()
    return save_dir


if __name__ == "__main__":
    run_batch_retrain()
