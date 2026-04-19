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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_Tuning")

# ---------------------------------------------------------------------------
# Objective priority weights for Pareto-front selection.
# NASA Asymmetric Score is the primary metric (safety-critical correctness).
# Model parameter count is the efficiency proxy (L1/L2 cache residency goal).
# RMSE is a useful secondary signal but not the primary concern.
# ---------------------------------------------------------------------------
PARETO_WEIGHTS = (0.65, 0.25, 0.10)   # (nasa_score, n_params, rmse)

# Parameter ceiling above which efficiency pressure kicks in (aligns with
# the <500KB / <1MB model size targets in ARCHITECTURE.md).
PARAM_BUDGET = 50_000

# In-trial early stopping: how many epochs without NASA improvement before
# we abandon a trial mid-run.  Replaces Optuna's pruner API which does not
# support multi-objective studies.
TRIAL_PATIENCE = 5

TUNING_EPOCHS = 20   # Bad trials exit early via TRIAL_PATIENCE; good ones run all 20.
SUBSET_RATIO  = 0.30 # More engines → better NASA signal during tuning.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_subset_engines(df: pd.DataFrame, ratio: float = 0.30, seed: int = 42) -> pd.DataFrame:
    """Deterministic engine sub-sampling for fast NAS trials."""
    rng = np.random.RandomState(seed)
    unit_ids = df["unit_id"].unique()
    subset_ids = rng.choice(unit_ids, size=max(1, int(len(unit_ids) * ratio)), replace=False)
    return df[df["unit_id"].isin(subset_ids)].copy()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single training epoch using the asymmetric loss (aligns training objective
    with the NASA scoring function — penalises late predictions more harshly).
    Gradient clipping is applied to prevent instability with asymmetric gradients.

    Note: the unused `criterion` parameter from the original signature has been
    removed; the loss function is fixed to `asymmetric_loss` here.
    """
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = asymmetric_loss(output, y)
        loss.backward()
        # Clip gradients — asymmetric loss can produce large gradients for
        # over-predictions (d_i >= 0 branch scales as e^(d/10)).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (rmse, nasa_score).

    MSELoss is used for RMSE so we have a cheap, stable auxiliary metric.
    NASA score is computed from raw predictions and is the primary signal.
    """
    mse_criterion = nn.MSELoss()
    model.eval()
    running_mse = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = mse_criterion(output, y)
            running_mse += loss.item() * x.size(0)
            all_preds.append(output.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    rmse = float(np.sqrt(running_mse / len(loader.dataset)))
    y_pred = np.concatenate(all_preds).flatten()
    y_true = np.concatenate(all_targets).flatten()
    nasa_score = float(nasa_asymmetric_score(y_pred, y_true))
    return rmse, nasa_score


def select_best_from_pareto(
    trials: List[optuna.trial.FrozenTrial],
    weights: Tuple[float, float, float] = PARETO_WEIGHTS,
) -> optuna.trial.FrozenTrial:
    """
    Select the single best trial from the NSGA-II Pareto front using
    min-max normalised weighted scoring.

    Objectives in order: (nasa_score [0], n_params [1], rmse [2]).
    All objectives are to be minimised.  Lower weighted score = better trial.
    """
    if not trials:
        raise ValueError("Pareto front is empty — no completed trials found.")

    values = np.array([t.values for t in trials], dtype=float)
    mins, maxs = values.min(axis=0), values.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)  # guard zero range
    normalised = (values - mins) / ranges
    scores = normalised @ np.array(weights)

    best_idx = int(np.argmin(scores))

    # Log the full Pareto front for transparency.
    logger.info("=== Pareto Front Summary ===")
    for rank, (trial, score) in enumerate(
        sorted(zip(trials, scores), key=lambda x: x[1])[:10], start=1
    ):
        nasa, params, rmse = trial.values
        logger.info(
            f"  [{rank:02d}] weighted={score:.4f} | "
            f"nasa={nasa:.2f}  n_params={int(params):,}  rmse={rmse:.4f}"
        )
    logger.info(f"Selected trial #{trials[best_idx].number} (weighted score {scores[best_idx]:.4f})")
    return trials[best_idx]


# ---------------------------------------------------------------------------
# NSGA-II Objective  (3 objectives: nasa_score, n_params, rmse)
# ---------------------------------------------------------------------------

class Objective:
    """
    Three-objective NAS callable for Optuna NSGA-II.

    Objectives (all minimise):
        [0] nasa_score  — Primary.  Asymmetric safety-critical score.
        [1] n_params    — Efficiency. Raw parameter count (proxy for model size
                          and cache footprint per ARCHITECTURE.md §4.2).
        [2] rmse        — Auxiliary.  Standard regression quality signal.

    Why separate objectives instead of a combined penalty term?
    ----------------------------------------------------------------
    NSGA-II tracks the full Pareto front, so the trade-off surface between
    accuracy and efficiency is explicit and navigable.  A penalty term
    collapses this surface into a single scalar and hides trade-offs that
    might be relevant at deployment time.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        device: torch.device,
    ):
        self.train_df    = train_df
        self.val_df      = val_df
        self.feature_cols = feature_cols
        self.device      = device

        # Pre-build loaders once per Objective instance (not per trial).
        # The engine subset is fixed at construction to ensure all trials
        # are evaluated on the same data distribution.
        subset_train = get_subset_engines(self.train_df, ratio=SUBSET_RATIO, seed=42)
        self._train_loader, self._val_loader, _ = get_dataloaders(
            subset_train, self.val_df, self.val_df,
            feature_cols=self.feature_cols,
            batch_size=64,
            seed=42,
        )

    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        # --- Architecture hyperparameters ---
        num_blocks  = trial.suggest_int("num_blocks",  1, 4)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        # BUG FIX: dilation was suggested but never forwarded to the model constructor.
        dilation    = trial.suggest_int("dilation", 1, 3)
        fc_units    = trial.suggest_categorical("fc_units", [16, 32, 64])

        out_channels_list: List[int]   = []
        use_bn_list:       List[bool]  = []
        dropout_list:      List[float] = []
        for i in range(num_blocks):
            out_channels_list.append(
                trial.suggest_categorical(f"out_channels_b{i}", [16, 32, 64])
            )
            use_bn_list.append(
                trial.suggest_categorical(f"use_bn_b{i}", [True, False])
            )
            dropout_list.append(
                trial.suggest_float(f"dropout_b{i}", 0.0, 0.5)
            )

        # --- Optimiser hyperparameters ---
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # --- Build model ---
        input_channels = len(self.feature_cols)
        model = RUL_1D_CNN(
            input_channels  = input_channels,
            num_blocks       = num_blocks,
            out_channels_list= out_channels_list,
            kernel_size      = kernel_size,
            dilation         = dilation,   # Fixed: now forwarded correctly.
            use_bn_list      = use_bn_list,
            dropout_list     = dropout_list,
            fc_units         = fc_units,
        ).to(self.device)

        n_params  = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # CosineAnnealingLR is well-suited for short tuning windows: it
        # provides a smooth, predictable decay that does not require
        # plateau detection over 20 epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TUNING_EPOCHS, eta_min=lr * 0.01
        )

        best_nasa          = float("inf")
        best_rmse          = float("inf")
        epochs_no_improve  = 0

        for epoch in range(TUNING_EPOCHS):
            train_one_epoch(model, self._train_loader, optimizer, self.device)
            val_rmse, val_nasa = evaluate(model, self._val_loader, self.device)
            scheduler.step()

            if val_nasa < best_nasa:
                best_nasa         = val_nasa
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if val_rmse < best_rmse:
                best_rmse = val_rmse

            # Manual early stopping: Optuna's trial.report() / should_prune()
            # API is NOT supported for multi-objective studies and raises
            # NotImplementedError.  We replicate the same behaviour ourselves:
            # abandon trials whose NASA score hasn't improved for TRIAL_PATIENCE
            # consecutive epochs, returning the best values seen so far.
            if epochs_no_improve >= TRIAL_PATIENCE:
                break

        return best_nasa, float(n_params), best_rmse


# ---------------------------------------------------------------------------
# Tuning Pipeline
# ---------------------------------------------------------------------------

def run_tuning_pipeline() -> Tuple[
    Dict[str, Any], pd.DataFrame, pd.DataFrame, CMAPSSPreprocessor, List[str]
]:
    """
    Full NAS/HPO tuning pipeline using NSGA-II multi-objective optimisation.

    Returns:
        best_params   — Hyperparameter dict for the selected Pareto-optimal trial.
        train_scaled  — Scaled training DataFrame (fitted preprocessor).
        val_scaled    — Scaled validation DataFrame (transform only).
        preprocessor  — Fitted CMAPSSPreprocessor instance.
        feature_cols  — Canonical feature column list from the fitted preprocessor.

    Why NSGA-II instead of TPE?
    ---------------------------
    TPE factorises the joint acquisition function over independent objectives,
    which loses the correlation structure of the (nasa, params, rmse) surface.
    NSGA-II uses non-dominated sorting + crowding distance to maintain a diverse
    Pareto front, making the accuracy/efficiency trade-off explicit and navigable.
    population_size=20 gives 5 effective generations over 100 trials, which is
    sufficient for the 3-objective surface at this search space dimensionality.
    """
    CMAPSS_DATA_DIR = os.path.join(DATA_DIR, "CMAPSSData")
    train_path = os.path.join(CMAPSS_DATA_DIR, "train_FD001.txt")

    if not os.path.exists(train_path):
        logger.error(f"Data not found at {train_path}")
        return None

    # --- Preprocessing ---
    preprocessor = CMAPSSPreprocessor(max_rul=125)
    raw_train    = preprocessor.load_data(train_path)
    raw_train    = preprocessor.add_piecewise_rul(raw_train)
    train_set, val_set = preprocessor.split_train_val_by_engine(raw_train, val_ratio=0.2)

    # Scaler fitted strictly on training data (zero temporal leakage).
    train_scaled = preprocessor.fit_transform(train_set)
    val_scaled   = preprocessor.transform(val_set)

    feature_cols: List[str] = preprocessor.active_features
    logger.info(f"Feature schema locked: {len(feature_cols)} active features.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Optuna study ---
    sampler = optuna.samplers.NSGAIISampler(
        seed=42,
        population_size=20,  # 100 trials / 20 = 5 generations.
    )
    # Note: Optuna's pruner API (trial.report / should_prune) is not supported
    # for multi-objective studies.  Early stopping is handled manually inside
    # the Objective.__call__ via a patience counter on NASA score.
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],  # nasa, n_params, rmse
        sampler=sampler,
    )

    objective = Objective(train_scaled, val_scaled, feature_cols, device)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # --- Select best trial from Pareto front ---
    pareto_trials = study.best_trials
    if not pareto_trials:
        logger.error("No completed trials found on Pareto front.")
        return None

    best_trial = select_best_from_pareto(pareto_trials, weights=PARETO_WEIGHTS)

    logger.info(
        f"Best trial #{best_trial.number}: "
        f"nasa={best_trial.values[0]:.2f}  "
        f"n_params={int(best_trial.values[1]):,}  "
        f"rmse={best_trial.values[2]:.4f}"
    )
    logger.info(f"Best hyperparameters: {best_trial.params}")

    return best_trial.params, train_scaled, val_scaled, preprocessor, feature_cols


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------

def finalize_model(
    best_params:  Dict[str, Any],
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    preprocessor: CMAPSSPreprocessor,
    feature_cols: List[str],
) -> str:
    """
    Retrain the final model with the selected hyperparameters on the full
    training set.  Bundles the ONNX model, fitted scaler, and feature schema
    into a single versioned directory under MODELS_DIR.

    Training regime:
        - OneCycleLR scheduler: better final convergence than ReduceLROnPlateau
          for a known epoch budget; max_lr drawn from the tuned lr value.
        - Early stopping on NASA score with patience=8 (increased from 5) since
          NASA score is noisier than RMSE and can improve after a plateau.
        - Best model state is saved and restored before ONNX export.

    Returns:
        save_dir: Path to the versioned model artifact directory.
    """
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_blocks   = best_params["num_blocks"]
    input_channels = len(feature_cols)

    out_channels_list = [best_params[f"out_channels_b{i}"] for i in range(num_blocks)]
    use_bn_list       = [best_params[f"use_bn_b{i}"]       for i in range(num_blocks)]
    dropout_list      = [best_params[f"dropout_b{i}"]      for i in range(num_blocks)]

    model = RUL_1D_CNN(
        input_channels   = input_channels,
        num_blocks       = num_blocks,
        out_channels_list= out_channels_list,
        kernel_size      = best_params["kernel_size"],
        dilation         = best_params["dilation"],   # Fixed: forwarded correctly.
        use_bn_list      = use_bn_list,
        dropout_list     = dropout_list,
        fc_units         = best_params["fc_units"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Final model: {n_params:,} parameters "
        f"({'within' if n_params <= PARAM_BUDGET else 'exceeds'} {PARAM_BUDGET:,} budget)"
    )

    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, val_df,
        feature_cols=feature_cols,
        batch_size=64,
        seed=42,
    )

    max_epochs = 70
    optimizer  = optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    # CosineAnnealingLR: smooth decay to eta_min over the full epoch budget.
    # Chosen over OneCycleLR because train_one_epoch does not step a scheduler
    # per batch, making OneCycleLR's batch-level design incompatible without a
    # refactor.  CosineAnnealingLR is epoch-level and steps correctly here.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=best_params["lr"] * 1e-3,
    )

    # Early stopping: NASA score is the sole stopping criterion.
    patience           = 8   # Increased: NASA score can plateau then improve.
    best_val_nasa      = float("inf")
    epochs_no_improve  = 0
    best_model_state   = None

    logger.info("Retraining final model on full training set...")
    for epoch in range(max_epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        val_rmse, nasa_score = evaluate(model, val_loader, device)
        scheduler.step()   # epoch-level step: must be called after evaluate()

        logger.info(
            f"Epoch {epoch + 1:03d}/{max_epochs} | "
            f"val_rmse={val_rmse:.4f}  nasa={nasa_score:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if nasa_score < best_val_nasa:
            best_val_nasa     = nasa_score
            epochs_no_improve = 0
            best_model_state  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(best nasa={best_val_nasa:.2f})."
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model state (nasa={best_val_nasa:.2f}).")

    # --- Versioned Artifact Bundle ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(MODELS_DIR, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.to("cpu")
    model_save_path = os.path.join(save_dir, "model.onnx")
    export_to_onnx(model, model_save_path, input_shape=(1, 30, input_channels))
    logger.info(f"Exported ONNX model to '{model_save_path}'.")

    preprocessor.save_artifacts(save_dir)

    logger.info(
        f"Artifact bundle complete → '{save_dir}'\n"
        f"  • model.onnx          ({os.path.getsize(model_save_path) / 1024:.1f} KB)\n"
        f"  • scaler.joblib\n"
        f"  • feature_schema.json\n"
        f"  Best val nasa_score: {best_val_nasa:.2f}"
    )
    return save_dir


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    result = run_tuning_pipeline()
    if result is not None:
        best_hparams, train_data, val_data, preprocessor, feature_cols = result
        finalize_model(best_hparams, train_data, val_data, preprocessor, feature_cols)