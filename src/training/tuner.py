import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Any, List

from src.models.architecture import RUL_1D_CNN
from src.data.loaders import get_dataloaders
from src.data.preprocess import CMAPSSPreprocessor
from src.core.config import set_seed, DATA_DIR
from src.training.utils import build_scaler, train_one_epoch
from src.training.eval import evaluate
from src.training.nas_utils import select_best_from_pareto

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
SUBSET_RATIO  = 0.30 # More engines -> better NASA signal during tuning.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_subset_engines(df: pd.DataFrame, ratio: float = 0.30, seed: int = 42) -> pd.DataFrame:
    """Deterministic engine sub-sampling for fast NAS trials."""
    rng = np.random.RandomState(seed)
    unit_ids = df["unit_id"].unique()
    subset_ids = rng.choice(unit_ids, size=max(1, int(len(unit_ids) * ratio)), replace=False)
    return df[df["unit_id"].isin(subset_ids)].copy()


# ---------------------------------------------------------------------------
# NSGA-II Objective  (3 objectives: nasa_score, n_params, rmse)
# ---------------------------------------------------------------------------

class Objective:
    """
    Three-objective NAS callable for Optuna NSGA-II.

    Objectives (all minimise):
        [0] nasa_score  -- Primary.  Asymmetric safety-critical score.
        [1] n_params    -- Efficiency. Raw parameter count (proxy for model size
                          and cache footprint per ARCHITECTURE.md .2).
        [2] rmse        -- Auxiliary.  Standard regression quality signal.

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
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # CosineAnnealingLR is well-suited for short tuning windows: it
        # provides a smooth, predictable decay that does not require
        # plateau detection over 20 epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TUNING_EPOCHS, eta_min=lr * 0.01
        )

        # One scaler per trial -- GradScaler maintains internal scale state
        # across epochs and must not be recreated inside the epoch loop.
        scaler = build_scaler(self.device)

        best_nasa          = float("inf")
        best_rmse          = float("inf")
        epochs_no_improve  = 0

        for epoch in range(TUNING_EPOCHS):
            train_one_epoch(model, self._train_loader, optimizer, self.device, scaler)
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
    Dict[str, Any], pd.DataFrame, pd.DataFrame, CMAPSSPreprocessor, List[str],
    optuna.Study, optuna.trial.FrozenTrial,
]:
    """
    Full NAS/HPO tuning pipeline using NSGA-II multi-objective optimisation.

    Returns:
        best_params   -- Hyperparameter dict for the selected Pareto-optimal trial.
        train_scaled  -- Scaled training DataFrame (fitted preprocessor).
        val_scaled    -- Scaled validation DataFrame (transform only).
        preprocessor  -- Fitted CMAPSSPreprocessor instance.
        feature_cols  -- Canonical feature column list from the fitted preprocessor.
        study         -- Completed Optuna study (passed to finalize_model for plot).
        best_trial    -- The FrozenTrial selected by select_best_from_pareto().

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
        population_size=50,  # 100 trials / 20 = 5 generations.
    )
    # Note: Optuna's pruner API (trial.report / should_prune) is not supported
    # for multi-objective studies.  Early stopping is handled manually inside
    # the Objective.__call__ via a patience counter on NASA score.
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],  # nasa, n_params, rmse
        sampler=sampler,
    )

    objective = Objective(train_scaled, val_scaled, feature_cols, device)
    study.optimize(objective, n_trials=300, show_progress_bar=True)

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

    return best_trial.params, train_scaled, val_scaled, preprocessor, feature_cols, study, best_trial


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    result = run_tuning_pipeline()
    if result is not None:
        best_hparams, train_data, val_data, preprocessor, feature_cols, study, best_trial = result
        from src.training.train import finalize_model
        finalize_model(best_hparams, train_data, val_data, preprocessor, feature_cols, study, best_trial)
