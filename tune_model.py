import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any
from model import RUL_1D_CNN, export_to_onnx
from data_loaders import get_dataloaders
from preprocess_cmapss import CMAPSSPreprocessor

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_Tuning")

def nasa_asymmetric_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculates the NASA Asymmetric Scoring Function.
    Penalizes late predictions more than early ones.
    """
    d = y_pred - y_true
    score = 0
    for val in d:
        if val < 0:
            score += np.exp(-val / 13.0) - 1
        else:
            score += np.exp(val / 10.0) - 1
    return float(score)

def asymmetric_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Stabilized NASA Asymmetric Loss for PyTorch.
    - Clamps error to prevent exponential overflow.
    - Ensures shape parity.
    """
    # Ensure shapes match to prevent broadcasting errors
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    d = y_pred - y_true
    
    # 1. Clamp d to a safe range (e.g., -50 to 50) 
    # This prevents exp(d) from reaching Infinity/NaN during early training
    d_clamped = torch.clamp(d, min=-65, max=50) 
    
    # 2. Apply asymmetric penalty
    loss = torch.where(
        d_clamped < 0, 
        torch.exp(-d_clamped / 13.0) - 1.0, 
        torch.exp(d_clamped / 10.0) - 1.0
    )
    
    return torch.mean(loss)

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
        loss = asymmetric_loss(output, y) # Use Asymmetric Loss for training
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
            loss = criterion(output, y) # Standard MSE for tracking
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
        # Hyperparameters
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        out_channels_base = trial.suggest_categorical("out_channels_base", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        use_bn = trial.suggest_categorical("use_bn", [True, False])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        out_channels_list = [out_channels_base] * num_blocks
        
        # DataLoaders for trial (subset)
        subset_train = get_subset_engines(self.train_df, ratio=0.2, seed=42)
        train_loader, val_loader, _ = get_dataloaders(subset_train, self.val_df, self.val_df, batch_size=64)
        
        # Determine actual input channels (features)
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
        best_val_nasa = float('inf')
        
        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, None, self.device)
            val_rmse, nasa_score = evaluate(model, val_loader, criterion, self.device)
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_nasa = nasa_score
        
        # Complexity Penalty: significant if > 100k
        penalty = 0.0
        if total_params > 100000:
            penalty = (total_params - 100000) * 0.01 
            
        return best_val_rmse + penalty, best_val_nasa

def run_tuning():
    # Setup data
    DATA_DIR = "data/CMAPSSData"
    train_path = os.path.join(DATA_DIR, "train_FD001.txt")
    
    if not os.path.exists(train_path):
        logger.error("Data not found. Please ensure FD001 data is in data/CMAPSSData")
        return

    preprocessor = CMAPSSPreprocessor(max_rul=125)
    raw_train = preprocessor.load_data(train_path)
    raw_train = preprocessor.add_piecewise_rul(raw_train)
    train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)
    train_scaled = preprocessor.fit_transform(train_set)
    val_scaled = preprocessor.transform(val_set)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Multi-objective study: Minimize RMSE and NASA Score
    study = optuna.create_study(directions=["minimize", "minimize"], pruner=optuna.pruners.MedianPruner())
    study.optimize(Objective(train_scaled, val_scaled, device), n_trials=100)
    
    # Pareto Front Selection
    trials = study.best_trials
    # Sort by RMSE (objective 0)
    trials.sort(key=lambda t: t.values[0])
    # Take top 5 with best RMSE
    top_candidates = trials[:5]
    # From top candidates, select the one with the lowest NASA score (objective 1)
    best_trial = min(top_candidates, key=lambda t: t.values[1])
    
    logger.info(f"Best trial from Pareto front: {best_trial.params}")
    logger.info(f"Best RMSE + Penalty: {best_trial.values[0]}, NASA Score: {best_trial.values[1]}")
    
    return best_trial.params, train_scaled, val_scaled

def finalize_model(best_params, train_df, val_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_blocks = best_params["num_blocks"]
    out_channels_base = best_params["out_channels_base"]
    out_channels_list = [out_channels_base] * num_blocks
    
    # Determine actual input channels (features)
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
    
    # Train on FULL training set
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, val_df, batch_size=64)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.MSELoss()
    
    # Implement LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Implement Early Stopping
    patience = 15
    best_val_nasa = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    max_epochs = 100
    
    logger.info("Retraining final model on full training set with Asymmetric Loss, LR Decay, and Early Stopping...")
    for epoch in range(max_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, None, device)
        
        # Evaluate every epoch
        val_rmse, nasa_score = evaluate(model, val_loader, criterion, device)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Val RMSE: {val_rmse:.4f}, NASA Score: {nasa_score:.2f}")
            
        # Step scheduler based on NASA score
        scheduler.step(nasa_score)
        
        # Check early stopping
        if nasa_score < best_val_nasa:
            best_val_nasa = nasa_score
            epochs_no_improve = 0
            best_model_state = model.state_dict() # type: ignore
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best NASA Score: {best_val_nasa:.2f}")
            break
            
    # Restore the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
            
    # Export to ONNX
    export_to_onnx(model.cpu(), "best_rul_model.onnx", input_shape=(1, 30, input_channels))
    logger.info("Final best model saved and exported.")

if __name__ == "__main__":
    best_hparams, train_data, val_data = run_tuning()
    if best_hparams:
        finalize_model(best_hparams, train_data, val_data)
