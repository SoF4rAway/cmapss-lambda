import logging
import os
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.core.config import set_seed, DATA_DIR, PROJECT_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CMAPSS_Preprocess")

class CMAPSSPreprocessor:
    """
    Production-ready preprocessor for NASA C-MAPSS dataset.
    """
    def __init__(self, max_rul: int = 130):
        self.max_rul = max_rul
        self.scaler = StandardScaler()
        self.constant_sensors: List[str] = []
        self.least_monotonic_sensors: List[str] = []
        self.dropped_features: List[str] = []
        self.surviving_features: List[str] = []
        self.sensor_cols: List[str] = []
        self.op_setting_cols: List[str] = []
        self.metadata_cols = ["unit_id", "cycle"]

    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        num_cols = df.shape[1]
        self.op_setting_cols = [f"op_setting_{i}" for i in range(1, 4)]
        self.sensor_cols = [f"sensor_{i}" for i in range(1, num_cols - len(self.metadata_cols) - len(self.op_setting_cols) + 1)]
        
        col_names = self.metadata_cols + self.op_setting_cols + self.sensor_cols
        df.columns = col_names
        
        logger.info(f"Loaded {file_path} with shape {df.shape}. Identified {len(self.sensor_cols)} sensors.")
        return df

    def add_piecewise_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        df["rul"] = max_cycle - df["cycle"]
        df["rul"] = df["rul"].clip(upper=self.max_rul)
        logger.info(f"Calculated piecewise RUL (clipped at {self.max_rul}).")
        return df

    def split_train_val_by_engine(self, df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unit_ids = df["unit_id"].unique()
        np.random.seed(seed)
        np.random.shuffle(unit_ids)
        
        val_size = int(len(unit_ids) * val_ratio)
        val_ids = unit_ids[:val_size]
        train_ids = unit_ids[val_size:]
        
        train_df = df[df["unit_id"].isin(train_ids)].copy()
        val_df = df[df["unit_id"].isin(val_ids)].copy()
        
        logger.info(f"Split data into {len(train_ids)} training and {len(val_ids)} validation engines.")
        return train_df, val_df

    def fit_transform(self, df: pd.DataFrame, variance_threshold: float = 1e-5, monotonicity_threshold: float = 0.2, drop_n_least_monotonic: int = 5) -> pd.DataFrame:
        features = self.op_setting_cols + self.sensor_cols
        variances = df[features].var()
        self.constant_sensors = list(variances[variances <= variance_threshold].index)
        active_features = [f for f in features if f not in self.constant_sensors]
        
        logger.info(f"Calculating monotonicity (Spearman) across {len(active_features)} active features...")
        monotonicity_scores = {}
        for feature in active_features:
            corrs = df.groupby("unit_id").apply(lambda g: g["cycle"].corr(g[feature], method="spearman"))
            avg_abs_corr = corrs.abs().mean()
            monotonicity_scores[feature] = avg_abs_corr
            
        ranked_monotonicity = sorted(monotonicity_scores.items(), key=lambda x: x[1])
        self.least_monotonic_sensors = [f for f, s in ranked_monotonicity[:drop_n_least_monotonic]]
        for f, s in ranked_monotonicity[drop_n_least_monotonic:]:
            if s < monotonicity_threshold:
                self.least_monotonic_sensors.append(f)

        self.dropped_features = list(set(self.constant_sensors + self.least_monotonic_sensors))
        self.surviving_features = [f for f in features if f not in self.dropped_features]
        
        logger.info(f"Feature Selection Results:")
        logger.info(f"  - Constant Sensors ({len(self.constant_sensors)}): {self.constant_sensors}")
        logger.info(f"  - Least Monotonic Sensors ({len(self.least_monotonic_sensors)}): {self.least_monotonic_sensors}")
        logger.info(f"  - Total Dropped: {len(self.dropped_features)}")
        logger.info(f"  - Surviving Features: {len(self.surviving_features)}")

        # Use PROJECT_ROOT for schema export
        schema_path = os.path.join(PROJECT_ROOT, "feature_schema.json")
        with open(schema_path, "w") as f:
            json.dump({"dropped_features": self.dropped_features}, f, indent=4)
        logger.info(f"Exported dropped features schema to {schema_path}")

        df_scaled = df.copy()
        df_scaled[self.surviving_features] = self.scaler.fit_transform(df[self.surviving_features])
        df_scaled = df_scaled.drop(columns=self.dropped_features)
        
        logger.info(f"Fitted scaler and transformed training data. Shape: {df_scaled.shape}")
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = df.copy()
        df_scaled[self.surviving_features] = self.scaler.transform(df[self.surviving_features])
        df_scaled = df_scaled.drop(columns=self.dropped_features)
        logger.info(f"Transformed data using existing scaler. Dropped {len(self.dropped_features)} features.")
        return df_scaled

    def load_and_label_test_data(self, test_file: str, rul_file: str) -> pd.DataFrame:
        test_df = self.load_data(test_file)
        true_ruls = pd.read_csv(rul_file, header=None).values.flatten()
        unit_ids = test_df["unit_id"].unique()
        
        if len(true_ruls) != len(unit_ids):
            raise ValueError(f"Mismatch between number of units ({len(unit_ids)}) and RUL labels ({len(true_ruls)})")

        rul_map = dict(zip(unit_ids, true_ruls))
        max_cycles = test_df.groupby("unit_id")["cycle"].transform("max")
        test_df["rul"] = test_df["unit_id"].map(rul_map) + (max_cycles - test_df["cycle"])
        test_df["rul"] = test_df["rul"].clip(upper=self.max_rul)
        
        logger.info(f"Loaded and labeled test set from {test_file} and {rul_file}.")
        return test_df

def run_preprocess_pipeline():
    set_seed(42)
    # Use DATA_DIR from config
    CMAPSS_DATA_DIR = os.path.join(DATA_DIR, "CMAPSSData")
    train_path = os.path.join(CMAPSS_DATA_DIR, "train_FD001.txt")
    test_path = os.path.join(CMAPSS_DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(CMAPSS_DATA_DIR, "RUL_FD001.txt")
    
    if os.path.exists(train_path):
        preprocessor = CMAPSSPreprocessor(max_rul=130)
        raw_train = preprocessor.load_data(train_path)
        raw_train = preprocessor.add_piecewise_rul(raw_train)
        train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)
        train_scaled = preprocessor.fit_transform(train_set)
        val_scaled = preprocessor.transform(val_set)
        
        if os.path.exists(test_path) and os.path.exists(rul_path):
            test_labeled = preprocessor.load_and_label_test_data(test_path, rul_path)
            test_scaled = preprocessor.transform(test_labeled)
            logger.info(f"Final Shapes - Train: {train_scaled.shape}, Val: {val_scaled.shape}, Test: {test_scaled.shape}")
    else:
        logger.warning(f"C-MAPSS data not found at {train_path}. Skipping demonstration.")

if __name__ == "__main__":
    run_preprocess_pipeline()
