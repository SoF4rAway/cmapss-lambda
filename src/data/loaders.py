import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List
from src.core.config import seed_worker

class CMAPSSTrainDataset(Dataset):
    """
    PyTorch Dataset for CMAPSS training and validation data using sliding windows.

    Feature columns are passed explicitly via feature_cols to eliminate any
    hardcoded guessing logic and ensure the schema is deterministic across
    train, validation, and test pipelines.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.features, self.labels = self._prepare_sequences(df)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        all_features = []
        all_labels = []

        for unit_id, group in df.groupby("unit_id"):
            group_data = group[self.feature_cols].values
            group_labels = group["rul"].values
            num_rows = len(group_data)
            if num_rows < self.sequence_length:
                continue
            for i in range(num_rows - self.sequence_length + 1):
                window = group_data[i : i + self.sequence_length]
                label = group_labels[i + self.sequence_length - 1]
                all_features.append(window)
                all_labels.append(label)

        return torch.tensor(np.array(all_features), dtype=torch.float32), \
               torch.tensor(np.array(all_labels), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CMAPSSTestDataset(Dataset):
    """
    PyTorch Dataset for CMAPSS test data extracting only the last sequence per engine.
    Implements pre-padding with zeros for short sequences.

    Feature columns are passed explicitly via feature_cols to ensure strict
    alignment with the training schema.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.features, self.labels = self._prepare_sequences(df)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        num_features = len(self.feature_cols)
        all_features = []
        all_labels = []
        unit_ids = df["unit_id"].unique()

        for unit_id in unit_ids:
            group = df[df["unit_id"] == unit_id]
            group_data = group[self.feature_cols].values
            group_labels = group["rul"].values
            num_rows = len(group_data)

            if num_rows >= self.sequence_length:
                window = group_data[-self.sequence_length:]
            else:
                padding_size = self.sequence_length - num_rows
                padding = np.zeros((padding_size, num_features))
                window = np.vstack((padding, group_data))

            label = group_labels[-1]
            all_features.append(window)
            all_labels.append(label)

        return torch.tensor(np.array(all_features), dtype=torch.float32), \
               torch.tensor(np.array(all_labels), dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrapper function to create PyTorch DataLoaders with reproducibility constraints.

    Args:
        train_df: Scaled training DataFrame.
        val_df: Scaled validation DataFrame.
        test_df: Scaled test DataFrame.
        feature_cols: Explicit, deterministic list of active feature columns derived
                      from CMAPSSPreprocessor.active_features.
        sequence_length: Sliding window size.
        batch_size: Batch size for all loaders.
        seed: Global seed for the torch Generator.
    """
    train_ds = CMAPSSTrainDataset(train_df, feature_cols, sequence_length)
    val_ds = CMAPSSTrainDataset(val_df, feature_cols, sequence_length)
    test_ds = CMAPSSTestDataset(test_df, feature_cols, sequence_length)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, val_loader, test_loader
