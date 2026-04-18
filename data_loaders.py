import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List

class CMAPSSTrainDataset(Dataset):
    """
    PyTorch Dataset for CMAPSS training and validation data using sliding windows.
    """
    def __init__(self, df: pd.DataFrame, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.features, self.labels = self._prepare_sequences(df)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_cols = [c for c in df.columns if c not in ["unit_id", "cycle", "rul"]]
        
        all_features = []
        all_labels = []
        
        for unit_id, group in df.groupby("unit_id"):
            group_data = group[feature_cols].values
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
    """
    def __init__(self, df: pd.DataFrame, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.features, self.labels = self._prepare_sequences(df)

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_cols = [c for c in df.columns if c not in ["unit_id", "cycle", "rul"]]
        num_features = len(feature_cols)
        
        all_features = []
        all_labels = []
        
        # Ensure we maintain engine order
        unit_ids = df["unit_id"].unique()
        
        for unit_id in unit_ids:
            group = df[df["unit_id"] == unit_id]
            group_data = group[feature_cols].values
            group_labels = group["rul"].values
            
            num_rows = len(group_data)
            
            if num_rows >= self.sequence_length:
                # Extract last sequence_length cycles
                window = group_data[-self.sequence_length:]
                label = group_labels[-1]
            else:
                # Pre-pad with zeros
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
    sequence_length: int = 30,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrapper function to create PyTorch DataLoaders.
    """
    train_ds = CMAPSSTrainDataset(train_df, sequence_length)
    val_ds = CMAPSSTrainDataset(val_df, sequence_length)
    test_ds = CMAPSSTestDataset(test_df, sequence_length)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
