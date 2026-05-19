"""
Module for generating realistic drifted C-MAPSS telemetry data.

This script reads the original NASA C-MAPSS test data and programmatically
injects realistic sensor degradation (sub-linear Gaussian noise calibrated per sensor) 
and environmental drift (linear scalar offset) to simulate field anomalies.
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


def generate_drift_data_realistic(
    input_filename: str = "test_FD001.txt",
    output_filename: str = "test_FD001_drifted.txt",
    noise_factor: float = 0.05,      # Noise tambahan maksimum 5% dari std asli sensor
    drift_factor: float = 0.01,      # Drift per siklus sebesar 1% dari std asli sensor
    drift_sensors: List[str] = None
) -> None:
    """
    Reads C-MAPSS data and injects realistic, sensor-scaled noise and drift.
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

    # Hitung standar deviasi asli tiap sensor untuk skala gangguan yang adil
    logger.info("Calculating baseline sensor volatilities...")
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    baseline_std = df[sensor_cols].std()

    logger.info("Injecting realistic sub-linear Gaussian noise...")
    # Menggunakan akar kuadrat waktu agar noise tidak meledak di siklus akhir (sub-linear)
    time_factor = np.sqrt(df["time_cycles"])
    
    for sensor_col in sensor_cols:
        orig_std = baseline_std[sensor_col]
        
        # Jika sensor konstan/mati (std = 0), lewati agar tidak merusak data konstannya
        if orig_std == 0:
            continue
            
        # Skala standar deviasi dinamis: proporsional terhadap karakteristik sensor itu sendiri
        dynamic_std = noise_factor * orig_std * time_factor
        
        noise = np.random.normal(
            loc=0.0,
            scale=dynamic_std,
            size=len(df)
        )
        df[sensor_col] += noise

    logger.info("Injecting realistic scaled linear drift into %s...", drift_sensors)
    # Drift juga disesuaikan dengan skala nilai masing-masing sensor terpilih
    for sensor_col in drift_sensors:
        if sensor_col in df.columns:
            orig_std = baseline_std[sensor_col]
            if orig_std == 0:
                orig_std = 1.0 # Fallback jika sensor bernilai konstan tapi ingin di-drift
                
            # Nilai drift bergeser secara linear berdasarkan standar deviasi sensor
            drift = drift_factor * orig_std * df["time_cycles"]
            df[sensor_col] += drift
        else:
            logger.warning("Sensor %s not found in dataframe.", sensor_col)

    logger.info("Saving realistic drifted data to %s...", output_path)
    df.to_csv(output_path, sep=" ", header=False, index=False)
    logger.info("Realistic drift data generation complete.")


if __name__ == "__main__":
    generate_drift_data_realistic()
