import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any
from datetime import datetime

from src.models.architecture import RUL_1D_CNN, export_to_onnx
from src.data.loaders import get_dataloaders
from src.data.preprocess import CMAPSSPreprocessor
from src.core.config import set_seed, DATA_DIR, MODELS_DIR
from src.training.utils import asymmetric_loss, nasa_asymmetric_score

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_Tuning")

def get_subset_engines(df: pd.DataFrame, ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic subsetting of engines.
    """
    unit_ids = df["unit_id"].unique()
    np.random.seed(seed)
    subset_ids = np.random.choice(unit_ids, size=int(len(unit_ids) * ratio), replace=False)
    return df[df["unit_id"].isin(subset_ids)].copy()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = asymmetric_loss(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            running_loss += loss.item() * x.size(0)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    mse = running_loss / len(loader.dataset)
    rmse = np.sqrt(mse)
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    nasa_score = nasa_asymmetric_score(y_pred, y_true)
    return rmse, nasa_score

class Objective:
    def __init__(self, train_df, val_df, device):
        self.train_df = train_df
        self.val_df = val_df
        self.device = device

    def __call__(self, trial):
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        out_channels_base = trial.suggest_categorical("out_channels_base", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        use_bn = trial.suggest_categorical("use_bn", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        out_channels_list = [out_channels_base] * num_blocks
        subset_train = get_subset_engines(self.train_df, ratio=0.2, seed=42)
        train_loader, val_loader, _ = get_dataloaders(subset_train, self.val_df, self.val_df, batch_size=64, seed=42)
        
        feature_cols = [c for c in self.train_df.columns if c not in ["unit_id", "cycle", "rul"]]
        input_channels = len(feature_cols)
        
        model = RUL_1D_CNN(
            input_channels=input_channels,
            num_blocks=num_blocks,
            out_channels_list=out_channels_list,
            kernel_size=kernel_size,
            use_bn=use_bn,
            dropout=dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        epochs = 20
        best_val_rmse = float('inf')
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, None, self.device)
            val_rmse, _ = evaluate(model, val_loader, criterion, self.device)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
        
        penalty = 0.0
        if total_params > 100000:
            penalty = (total_params - 100000) * 0.01 
            
        return best_val_rmse + penalty, 0.0 # Placeholder for 2nd objective if needed

def run_tuning_pipeline():
    CMAPSS_DATA_DIR = os.path.join(DATA_DIR, "CMAPSSData")
    train_path = os.path.join(CMAPSS_DATA_DIR, "train_FD001.txt")
    
    if not os.path.exists(train_path):
        logger.error(f"Data not found at {train_path}")
        return

    preprocessor = CMAPSSPreprocessor(max_rul=125)
    raw_train = preprocessor.load_data(train_path)
    raw_train = preprocessor.add_piecewise_rul(raw_train)
    train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)
    train_scaled = preprocessor.fit_transform(train_set)
    val_scaled = preprocessor.transform(val_set)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler, pruner=optuna.pruners.MedianPruner())
    study.optimize(Objective(train_scaled, val_scaled, device), n_trials=100)
    
    trials = study.best_trials
    trials.sort(key=lambda t: t.values[0])
    best_trial = trials[0]
    
    logger.info(f"Best trial params: {best_trial.params}")
    return best_trial.params, train_scaled, val_scaled

def finalize_model(best_params, train_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = best_params["num_blocks"]
    out_channels_list = [best_params["out_channels_base"]] * num_blocks
    feature_cols = [c for c in train_df.columns if c not in ["unit_id", "cycle", "rul"]]
    input_channels = len(feature_cols)
    
    model = RUL_1D_CNN(
        input_channels=input_channels,
        num_blocks=num_blocks,
        out_channels_list=out_channels_list,
        kernel_size=best_params["kernel_size"],
        use_bn=best_params["use_bn"],
        dropout=best_params["dropout"]
    ).to(device)
    
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, val_df, batch_size=64, seed=42)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    patience = 15
    best_val_nasa = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    max_epochs = 100
    
    logger.info("Retraining final model...")
    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, None, device)
        val_rmse, nasa_score = evaluate(model, val_loader, criterion, device)
        scheduler.step(nasa_score)
        
        if nasa_score < best_val_nasa:
            best_val_nasa = nasa_score
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(MODELS_DIR, f"{timestamp}_rul_model.onnx")
    export_to_onnx(model.cpu(), model_save_path, input_shape=(1, 30, input_channels))
    logger.info(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    set_seed(42)
    best_hparams, train_data, val_data = run_tuning_pipeline()
    if best_hparams:
        finalize_model(best_hparams, train_data, val_data)
