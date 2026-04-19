import logging
import os
import json
from typing import List, Tuple, Optional

import joblib
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

    Supports two operational modes:
    - Fit Mode (artifact_dir=None): Calculates monotonicity, fits the scaler,
      and determines the final active_features list. Artifacts must be saved
      explicitly via save_artifacts().
    - Transform Mode (artifact_dir provided): Completely skips monotonicity
      calculation. Loads feature_schema.json and scaler.joblib from the given
      directory and applies the transform deterministically.
    """
    def __init__(self, max_rul: int = 130, artifact_dir: Optional[str] = None):
        self.max_rul = max_rul
        self.scaler = StandardScaler()
        self.constant_sensors: List[str] = []
        self.least_monotonic_sensors: List[str] = []
        self.dropped_features: List[str] = []
        self.surviving_features: List[str] = []
        self.active_features: List[str] = []
        self.sensor_cols: List[str] = []
        self.op_setting_cols: List[str] = []
        self.metadata_cols = ["unit_id", "cycle"]
        self.artifact_dir: Optional[str] = artifact_dir

        if artifact_dir is not None:
            self._load_artifacts(artifact_dir)

    def _load_artifacts(self, artifact_dir: str) -> None:
        """
        Transform Mode: Load feature schema and scaler from a versioned artifact directory.
        Raises FileNotFoundError if required artifact files are missing.
        """
        schema_path = os.path.join(artifact_dir, "feature_schema.json")
        scaler_path = os.path.join(artifact_dir, "scaler.joblib")

        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"feature_schema.json not found at: {schema_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"scaler.joblib not found at: {scaler_path}")

        with open(schema_path, "r") as f:
            schema = json.load(f)
        self.active_features = schema["active_features"]
        self.surviving_features = self.active_features  # Alias for transform compatibility

        self.scaler = joblib.load(scaler_path)
        logger.info(
            f"Transform Mode: Loaded {len(self.active_features)} active features "
            f"and scaler from '{artifact_dir}'."
        )

    @property
    def get_active_features(self) -> List[str]:
        """Returns the deterministic list of active features after fit or load."""
        return self.active_features

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

    def fit_transform(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 1e-5,
        monotonicity_threshold: float = 0.2,
        drop_n_least_monotonic: int = 5
    ) -> pd.DataFrame:
        """
        Fit Mode: Calculates variance pruning, monotonicity selection, and fits the
        StandardScaler strictly on training data. Populates self.active_features.
        Must only be called on training data to ensure zero temporal leakage.
        """
        if self.artifact_dir is not None:
            raise RuntimeError(
                "fit_transform() cannot be called in Transform Mode (artifact_dir is set). "
                "Use transform() instead."
            )

        features = self.op_setting_cols + self.sensor_cols
        variances = df[features].var()
        self.constant_sensors = list(variances[variances <= variance_threshold].index)
        active_features = [f for f in features if f not in self.constant_sensors]

        logger.info(f"Calculating monotonicity (Spearman) across {len(active_features)} active features...")
        monotonicity_scores = {}
        for feature in active_features:
            corrs = df.groupby("unit_id").apply(
                lambda g: g["cycle"].corr(g[feature], method="spearman"),
                include_groups=False
            )
            avg_abs_corr = corrs.abs().mean()
            monotonicity_scores[feature] = avg_abs_corr

        ranked_monotonicity = sorted(monotonicity_scores.items(), key=lambda x: x[1])
        self.least_monotonic_sensors = [f for f, s in ranked_monotonicity[:drop_n_least_monotonic]]
        for f, s in ranked_monotonicity[drop_n_least_monotonic:]:
            if s < monotonicity_threshold:
                self.least_monotonic_sensors.append(f)

        self.dropped_features = list(set(self.constant_sensors + self.least_monotonic_sensors))
        self.surviving_features = [f for f in features if f not in self.dropped_features]

        # Canonical, deterministic feature list — order preserved from original feature list
        self.active_features = self.surviving_features

        logger.info(f"Feature Selection Results:")
        logger.info(f"  - Constant Sensors ({len(self.constant_sensors)}): {self.constant_sensors}")
        logger.info(f"  - Least Monotonic Sensors ({len(self.least_monotonic_sensors)}): {self.least_monotonic_sensors}")
        logger.info(f"  - Total Dropped: {len(self.dropped_features)}")
        logger.info(f"  - Active Features ({len(self.active_features)}): {self.active_features}")

        df_scaled = df.copy()
        df_scaled[self.active_features] = self.scaler.fit_transform(df[self.active_features])
        df_scaled = df_scaled.drop(columns=self.dropped_features)

        logger.info(f"Fitted scaler and transformed training data. Shape: {df_scaled.shape}")
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted/loaded scaler to a DataFrame, strictly sliced to active_features
        plus required metadata and label columns.
        """
        if not self.active_features:
            raise RuntimeError(
                "active_features is empty. Call fit_transform() first, or provide "
                "artifact_dir in __init__ to load a saved schema."
            )

        retain_cols = self.metadata_cols + ["rul"] if "rul" in df.columns else self.metadata_cols
        cols_to_keep = retain_cols + self.active_features
        # Gracefully handle columns that may not be in the dataframe
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]

        df_sliced = df[cols_to_keep].copy()
        df_sliced[self.active_features] = self.scaler.transform(df_sliced[self.active_features])

        logger.info(
            f"Transformed data using {'loaded' if self.artifact_dir else 'fitted'} scaler. "
            f"Retained {len(self.active_features)} active features. Shape: {df_sliced.shape}"
        )
        return df_sliced

    def save_artifacts(self, save_dir: str) -> None:
        """
        Persist scaler and feature schema into a versioned artifact directory.
        This method should be called after fit_transform() to bundle the artifacts
        alongside the ONNX model in the same timestamped directory.
        """
        if not self.active_features:
            raise RuntimeError(
                "Cannot save artifacts: active_features is empty. "
                "Call fit_transform() before save_artifacts()."
            )

        os.makedirs(save_dir, exist_ok=True)

        scaler_path = os.path.join(save_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to '{scaler_path}'.")

        schema_path = os.path.join(save_dir, "feature_schema.json")
        schema = {"active_features": self.active_features}
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=4)
        logger.info(f"Saved feature schema ({len(self.active_features)} features) to '{schema_path}'.")

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
