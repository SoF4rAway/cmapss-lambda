import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any, List
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
    Deterministic subsetting of engines for faster NAS trials.
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
    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        device: torch.device
    ):
        self.train_df = train_df
        self.val_df = val_df
        self.feature_cols = feature_cols
        self.device = device

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float]:
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        dilation = trial.suggest_int("dilation", 1, 3)
        fc_units = trial.suggest_categorical("fc_units", [16, 32, 64])

        out_channels_list = []
        use_bn_list = []
        dropout_list = []
        for i in range(num_blocks):
            out_channels_list.append(trial.suggest_categorical(f"out_channels_b{i}", [16, 32, 64]))
            use_bn_list.append(trial.suggest_categorical(f"use_bn_b{i}", [True, False]))
            dropout_list.append(trial.suggest_float(f"dropout_b{i}", 0.0, 0.5))

        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # Subsampling for speed during tuning
        subset_train = get_subset_engines(self.train_df, ratio=0.2, seed=42)
        train_loader, val_loader, _ = get_dataloaders(
            subset_train, self.val_df, self.val_df,
            feature_cols=self.feature_cols,
            batch_size=64,
            seed=42
        )

        input_channels = len(self.feature_cols)

        model = RUL_1D_CNN(
            input_channels=input_channels,
            num_blocks=num_blocks,
            out_channels_list=out_channels_list,
            kernel_size=kernel_size,
            use_bn_list=use_bn_list,
            dropout_list=dropout_list,
            fc_units=fc_units
        ).to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        epochs = 15  # Reduced for tuning speed
        best_val_rmse = float("inf")
        best_val_nasa = float("inf")

        for epoch in range(epochs):
            train_one_epoch(model, train_loader, optimizer, None, self.device)
            val_rmse, val_nasa = evaluate(model, val_loader, criterion, self.device)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
            if val_nasa < best_val_nasa:
                best_val_nasa = val_nasa

        # Complexity Penalty for hardware constraints
        penalty = 0.0
        if total_params > 50000:
            penalty = (total_params - 50000) * 0.001

        return best_val_rmse + penalty, best_val_nasa

def run_tuning_pipeline() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, CMAPSSPreprocessor, List[str]]:
    """
    Full NAS tuning pipeline. Returns the best hyperparameters, scaled DataFrames,
    the fitted preprocessor object, and the canonical feature column list.
    """
    CMAPSS_DATA_DIR = os.path.join(DATA_DIR, "CMAPSSData")
    train_path = os.path.join(CMAPSS_DATA_DIR, "train_FD001.txt")

    if not os.path.exists(train_path):
        logger.error(f"Data not found at {train_path}")
        return

    # Align with ARCHITECTURE.md (Piecewise RUL capped at 125)
    preprocessor = CMAPSSPreprocessor(max_rul=125)
    raw_train = preprocessor.load_data(train_path)
    raw_train = preprocessor.add_piecewise_rul(raw_train)
    train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)

    # Fit Mode: scaler fitted strictly on training data to ensure zero temporal leakage
    train_scaled = preprocessor.fit_transform(train_set)
    val_scaled = preprocessor.transform(val_set)

    # Retrieve the canonical, deterministic feature list from the fitted preprocessor
    feature_cols: List[str] = preprocessor.active_features
    logger.info(f"Feature schema locked: {len(feature_cols)} active features.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    sampler = optuna.samplers.TPESampler(seed=42)
    # Multi-objective study as per ARCHITECTURE.md
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(
        Objective(train_scaled, val_scaled, feature_cols, device),
        n_trials=5
    )

    # Select best trial based on primary objective (RMSE + Penalty)
    trials = study.best_trials
    trials.sort(key=lambda t: t.values[0])
    best_trial = trials[0]

    logger.info(f"Best trial params: {best_trial.params}")
    return best_trial.params, train_scaled, val_scaled, preprocessor, feature_cols

def finalize_model(
    best_params: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: CMAPSSPreprocessor,
    feature_cols: List[str]
) -> str:
    """
    Retrain the final model with best hyperparameters on the full training set.
    Bundles the ONNX model, fitted scaler, and feature schema into a single
    versioned directory under MODELS_DIR.

    Returns:
        save_dir: The path to the versioned model artifact directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks = best_params["num_blocks"]

    out_channels_list = [best_params[f"out_channels_b{i}"] for i in range(num_blocks)]
    use_bn_list = [best_params[f"use_bn_b{i}"] for i in range(num_blocks)]
    dropout_list = [best_params[f"dropout_b{i}"] for i in range(num_blocks)]

    input_channels = len(feature_cols)

    model = RUL_1D_CNN(
        input_channels=input_channels,
        num_blocks=num_blocks,
        out_channels_list=out_channels_list,
        kernel_size=best_params["kernel_size"],
        use_bn_list=use_bn_list,
        dropout_list=dropout_list,
        fc_units=best_params["fc_units"]
    ).to(device)

    # Retrain on full dataset with passed feature_cols
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, val_df,
        feature_cols=feature_cols,
        batch_size=64,
        seed=42
    )
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    patience = 5
    best_val_nasa = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    max_epochs = 70

    logger.info("Retraining final model on full training set...")
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
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # --- Versioned Artifact Bundle ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(MODELS_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Export ONNX model into the versioned directory
    model.eval()
    model.to("cpu")
    model_save_path = os.path.join(save_dir, "model.onnx")
    export_to_onnx(model, model_save_path, input_shape=(1, 30, input_channels))
    logger.info(f"Exported ONNX model to '{model_save_path}'.")

    # Save scaler.joblib and feature_schema.json alongside the model
    preprocessor.save_artifacts(save_dir)

    logger.info(
        f"Artifact bundle complete. All artifacts saved to: '{save_dir}'\n"
        f"  - model.onnx\n"
        f"  - scaler.joblib\n"
        f"  - feature_schema.json"
    )
    return save_dir

if __name__ == "__main__":
    set_seed(42)
    result = run_tuning_pipeline()
    if result is not None:
        best_hparams, train_data, val_data, preprocessor, feature_cols = result
        finalize_model(best_hparams, train_data, val_data, preprocessor, feature_cols)
