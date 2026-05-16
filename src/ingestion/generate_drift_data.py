"""
Module for generating drifted C-MAPSS telemetry data.

This script reads the original NASA C-MAPSS test data and programmatically
injects sensor degradation (progressive Gaussian noise) and environmental
drift (linear scalar offset) to simulate field anomalies.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reproducibility: Explicitly seed the generator
np.random.seed(42)


def generate_drift_data(
    input_filename: str = "test_FD001.txt",
    output_filename: str = "test_FD001_drifted.txt",
    noise_scale: float = 0.001,
    drift_scale: float = 0.01,
    drift_sensors: List[str] = None
) -> None:
    """
    Reads C-MAPSS data and injects noise and drift.

    Args:
        input_filename: Name of the original NASA C-MAPSS file.
        output_filename: Name of the file to save the modified data.
        noise_scale: Factor scaling Gaussian noise variance with time_cycles.
        drift_scale: Factor scaling linear drift with time_cycles.
        drift_sensors: List of sensor names to apply linear drift to.
    """
    if drift_sensors is None:
        drift_sensors = ["sensor_2", "sensor_11", "sensor_15"]

    base_path = Path("data/CMAPSSData")
    input_path = base_path / input_filename
    output_path = base_path / output_filename

    if not input_path.exists():
        logger.error("Input file not found at %s", input_path)
        return

    # Define column schema
    columns = [
        "unit_id", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]

    logger.info("Reading raw data from %s...", input_path)
    # The original NASA files are space-separated and often have trailing spaces
    # resulting in an extra NaN column if not handled.
    try:
        df = pd.read_csv(
            input_path,
            sep=r"\s+",
            header=None,
            names=columns,
            usecols=range(26)
        )
    except Exception as e:
        logger.error("Failed to read data: %s", e)
        return

    logger.info("Injecting progressive Gaussian noise (scale=%s)...", noise_scale)
    # Apply noise to all sensor columns (5 to 25)
    for i in range(1, 22):
        sensor_col = f"sensor_{i}"
        # Standard deviation scales with time_cycles: std = noise_scale * time_cycles
        noise = np.random.normal(
            loc=0.0,
            scale=noise_scale * df["time_cycles"],
            size=len(df)
        )
        df[sensor_col] += noise

    logger.info(
        "Injecting linear scalar drift into %s (scale=%s)...",
        drift_sensors,
        drift_scale
    )
    # Apply linear drift: value = value + (drift_scale * time_cycles)
    drift = drift_scale * df["time_cycles"]
    for sensor_col in drift_sensors:
        if sensor_col in df.columns:
            df[sensor_col] += drift
        else:
            logger.warning("Sensor %s not found in dataframe.", sensor_col)

    logger.info("Saving drifted data to %s...", output_path)
    # Maintain original format: space-separated, no headers, no index.
    df.to_csv(output_path, sep=" ", header=False, index=False)
    logger.info("Drift data generation complete.")


if __name__ == "__main__":
    generate_drift_data()
