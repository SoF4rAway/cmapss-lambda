import logging
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CMAPSS_Preprocess")

class CMAPSSPreprocessor:
    """
    Production-ready preprocessor for NASA C-MAPSS dataset.
    
    Handles dynamic column labeling, piecewise linear RUL calculation,
    leakage-free engine-based splitting, and robust scaling.
    """

    def __init__(self, max_rul: int = 130):
        """
        Initialize the preprocessor.

        Args:
            max_rul: Maximum RUL value to clip the target variable.
        """
        self.max_rul = max_rul
        self.scaler = StandardScaler()
        self.constant_sensors: List[str] = []
        self.sensor_cols: List[str] = []
        self.op_setting_cols: List[str] = []
        self.metadata_cols = ["unit_id", "cycle"]

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads C-MAPSS space-separated text files and dynamically labels columns.

        Args:
            file_path: Path to the .txt file.

        Returns:
            pd.DataFrame: Labeled dataframe.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read the file
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        
        # Identify columns
        # First two columns are always unit_id and cycle
        # Architecture says 3 op settings and 21 sensors (Total 26)
        # But we handle dynamically if needed
        num_cols = df.shape[1]
        self.op_setting_cols = [f"op_setting_{i}" for i in range(1, 4)]
        self.sensor_cols = [f"sensor_{i}" for i in range(1, num_cols - len(self.metadata_cols) - len(self.op_setting_cols) + 1)]
        
        col_names = self.metadata_cols + self.op_setting_cols + self.sensor_cols
        df.columns = col_names
        
        logger.info(f"Loaded {file_path} with shape {df.shape}. Identified {len(self.sensor_cols)} sensors.")
        return df

    def add_piecewise_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds Piecewise Linear RUL to the dataframe.

        Args:
            df: Input dataframe.

        Returns:
            pd.DataFrame: Dataframe with 'rul' column.
        """
        # Calculate max cycle per unit
        max_cycle = df.groupby("unit_id")["cycle"].transform("max")
        
        # Raw RUL (distance to failure)
        df["rul"] = max_cycle - df["cycle"]
        
        # Apply piecewise clipping
        df["rul"] = df["rul"].clip(upper=self.max_rul)
        
        logger.info(f"Calculated piecewise RUL (clipped at {self.max_rul}).")
        return df

    def split_train_val_by_engine(self, df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data by engine ID to ensure zero temporal leakage.

        Args:
            df: Input dataframe.
            val_ratio: Proportion of engines for validation.
            seed: Random seed for reproducibility.

        Returns:
            Tuple: (train_df, val_df)
        """
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

    def fit_transform(self, df: pd.DataFrame, threshold: float = 1e-5) -> pd.DataFrame:
        """
        Identifies constant sensors, fits scaler on training data, and transforms.

        Args:
            df: Training dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        # Feature columns (Op settings + Sensors)
        features = self.op_setting_cols + self.sensor_cols
        
        # Identify constant features (zero variance)
        variances = df[features].var()
        self.constant_sensors = list(variances[variances <= threshold].index)
        
        if self.constant_sensors:
            logger.info(f"Dropping constant sensors: {self.constant_sensors}")
        
        active_features = [f for f in features if f not in self.constant_sensors]
        
        # Fit and transform
        df_scaled = df.copy()
        df_scaled[active_features] = self.scaler.fit_transform(df[active_features])
        
        # Remove constant features
        df_scaled = df_scaled.drop(columns=self.constant_sensors)
        
        logger.info(f"Fitted scaler and transformed training data. Removed {len(self.constant_sensors)} constant columns.")
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms validation/test data using the fitted parameters.

        Args:
            df: Dataframe to transform.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        features = self.op_setting_cols + self.sensor_cols
        active_features = [f for f in features if f not in self.constant_sensors]
        
        df_scaled = df.copy()
        df_scaled[active_features] = self.scaler.transform(df[active_features])
        
        # Remove same constant features
        df_scaled = df_scaled.drop(columns=self.constant_sensors)
        
        logger.info("Transformed data using existing scaler.")
        return df_scaled

    def load_and_label_test_data(self, test_file: str, rul_file: str) -> pd.DataFrame:
        """
        Loads official test data and merges with true RUL labels.
        The test set ends before failure; RUL_FD00x.txt provides the RUL at the last cycle.

        Args:
            test_file: Path to test_FD00x.txt.
            rul_file: Path to RUL_FD00x.txt.

        Returns:
            pd.DataFrame: Labeled test dataframe.
        """
        # Load test set
        test_df = self.load_data(test_file)
        
        # Load true RUL labels (one per unit)
        true_ruls = pd.read_csv(rul_file, header=None).values.flatten()
        unit_ids = test_df["unit_id"].unique()
        
        if len(true_ruls) != len(unit_ids):
            raise ValueError(f"Mismatch between number of units ({len(unit_ids)}) and RUL labels ({len(true_ruls)})")

        # Map true RUL to the last cycle of each unit
        rul_map = dict(zip(unit_ids, true_ruls))
        
        # Calculate piecewise RUL for test trajectories
        # 1. Get max cycle per unit in test set
        max_cycles = test_df.groupby("unit_id")["cycle"].transform("max")
        
        # 2. RUL at cycle 'c' = (True RUL at last cycle) + (Max Cycle - Current Cycle)
        # Example: if true RUL at cycle 100 is 50, then at cycle 90 it was 60.
        test_df["rul"] = test_df["unit_id"].map(rul_map) + (max_cycles - test_df["cycle"])
        
        # 3. Clip piecewise
        test_df["rul"] = test_df["rul"].clip(upper=self.max_rul)
        
        logger.info(f"Loaded and labeled test set from {test_file} and {rul_file}.")
        return test_df

if __name__ == "__main__":
    # Example usage for FD001
    DATA_DIR = "data/CMAPSSData"
    train_path = os.path.join(DATA_DIR, "train_FD001.txt")
    test_path = os.path.join(DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    
    if os.path.exists(train_path):
        preprocessor = CMAPSSPreprocessor(max_rul=130)
        
        # 1. Load and add RUL
        raw_train = preprocessor.load_data(train_path)
        raw_train = preprocessor.add_piecewise_rul(raw_train)
        
        # 2. Split (leakage-free)
        train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)
        
        # 3. Scale
        train_scaled = preprocessor.fit_transform(train_set)
        val_scaled = preprocessor.transform(val_set)
        
        # 4. Process official test set
        if os.path.exists(test_path) and os.path.exists(rul_path):
            test_labeled = preprocessor.load_and_label_test_data(test_path, rul_path)
            test_scaled = preprocessor.transform(test_labeled)
            
            logger.info(f"Final Shapes - Train: {train_scaled.shape}, Val: {val_scaled.shape}, Test: {test_scaled.shape}")
            logger.info(f"Metadata check: {train_scaled.columns[:5].tolist()}")
    else:
        logger.warning("C-MAPSS data not found. Skipping demonstration.")
