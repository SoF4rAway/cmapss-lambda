import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import pandas as pd
import numpy as np
import os
import logging
import matplotlib
matplotlib.use("Agg")   # headless — no display required on training server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
# AMP Utilities
# ---------------------------------------------------------------------------

def build_scaler(device: torch.device) -> torch.amp.GradScaler:
    """
    Return a GradScaler appropriate for the given device.

    CUDA  — enabled fp16 scaler.  Scales the loss before backward to prevent
            fp16 underflow, then unscales before the optimizer step.
    CPU   — disabled (no-op) scaler.  CPU autocast uses bfloat16 which has the
            same exponent range as fp32 and does not need loss scaling.
    """
    return torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))


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
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    torch.amp.GradScaler,
) -> float:
    """
    Single training epoch with Automatic Mixed Precision (AMP).

    AMP strategy
    ------------
    Forward pass and loss computation run inside torch.amp.autocast, which
    dispatches eligible ops to fp16 (CUDA) or bf16 (CPU) for throughput.
    The GradScaler prevents fp16 underflow by scaling the loss before backward
    and unscaling before the optimizer step.

    Gradient clipping order matters: clip_grad_norm_ must be called AFTER
    scaler.unscale_() so it operates on the true (unscaled) gradient magnitudes.
    Clipping scaled gradients would produce norms ~1000× too large and silently
    corrupt training.

    On CPU the scaler is a no-op (enabled=False), so this function is equally
    correct for both CUDA and CPU execution paths.
    """
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type):
            output = model(x)
            loss   = asymmetric_loss(output, y)

        scaler.scale(loss).backward()

        # Unscale before clipping so clip_grad_norm_ sees true gradient norms.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

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
            # autocast for eval: matches training precision, avoids
            # redundant dtype conversion on the hot path.
            with torch.amp.autocast(device.type):
                output = model(x)
                loss   = mse_criterion(output, y)
            running_mse += loss.item() * x.size(0)
            all_preds.append(output.float().cpu().numpy())
            all_targets.append(y.float().cpu().numpy())

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
# Pareto Front Visualisation
# ---------------------------------------------------------------------------

def plot_pareto_front(
    study:      optuna.Study,
    best_trial: optuna.trial.FrozenTrial,
    save_dir:   str,
) -> str:
    """
    Render a 3-panel Pareto front scatter plot and save it to save_dir.

    Layout (1 row × 3 columns):
        [0] NASA Score  vs  Parameter Count   — primary accuracy/efficiency plane
        [1] NASA Score  vs  RMSE              — accuracy decomposition
        [2] Parameter Count  vs  RMSE         — efficiency vs auxiliary accuracy

    Visual encoding:
        • All completed trials      — grey circles, alpha=0.35
        • Pareto-front trials       — filled circles, coloured by RMSE (viridis_r)
        • Selected best trial       — red star (★), annotated with trial number

    Args:
        study:      Completed Optuna multi-objective study.
        best_trial: The trial selected by select_best_from_pareto().
        save_dir:   Directory where pareto_front.png is written.

    Returns:
        Path to the saved PNG file.
    """
    # ------------------------------------------------------------------ data
    all_trials    = [t for t in study.trials if t.values is not None]
    pareto_trials = study.best_trials

    def _extract(trials):
        nasa   = np.array([t.values[0] for t in trials])
        params = np.array([t.values[1] for t in trials])
        rmse   = np.array([t.values[2] for t in trials])
        return nasa, params, rmse

    all_nasa,    all_params,    all_rmse    = _extract(all_trials)
    front_nasa,  front_params,  front_rmse  = _extract(pareto_trials)
    sel_nasa  = best_trial.values[0]
    sel_params = best_trial.values[1]
    sel_rmse  = best_trial.values[2]

    # ------------------------------------------------------------------ Academic Style
    BG        = "#ffffff" 
    GRID_CLR  = "#dddddd"
    TEXT_CLR  = "#000000" 
    GREY_DOT  = "#7f7f7f" 
    SEL_CLR   = "#d62728" 
    CMAP      = "viridis" 

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    "#000000",
        "axes.labelcolor":   TEXT_CLR,
        "axes.labelsize":    10,       
        "xtick.color":       TEXT_CLR,
        "ytick.color":       TEXT_CLR,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "text.color":        TEXT_CLR,
        "grid.color":        GRID_CLR,
        "grid.linestyle":    "-",       
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "font.family":       "serif",  
        "legend.frameon":    True,      
        "legend.edgecolor":  "#000000",
        "savefig.dpi":       300,       
        "savefig.bbox":      "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"NSGA-II Pareto Front  ·  {len(pareto_trials)} non-dominated trials"
        f"  /  {len(all_trials)} total",
        fontsize=13, fontweight="bold", color=TEXT_CLR, y=1.01,
    )

    # Shared colormap normalisation across all three panels (RMSE range)
    rmse_min = min(all_rmse.min(), sel_rmse)
    rmse_max = max(all_rmse.max(), sel_rmse)
    norm     = plt.Normalize(vmin=rmse_min, vmax=rmse_max)
    sm       = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    # Panel definitions: (ax, x_data_all, y_data_all, x_front, y_front, sel_x, sel_y, xlabel, ylabel)
    panels = [
        (
            axes[0],
            all_params,   all_nasa,
            front_params, front_nasa,
            sel_params,   sel_nasa,
            "Parameter Count",  "NASA Asymmetric Score",
        ),
        (
            axes[1],
            all_rmse,   all_nasa,
            front_rmse, front_nasa,
            sel_rmse,   sel_nasa,
            "RMSE",             "NASA Asymmetric Score",
        ),
        (
            axes[2],
            all_params,   all_rmse,
            front_params, front_rmse,
            sel_params,   sel_rmse,
            "Parameter Count",  "RMSE",
        ),
    ]

    for ax, ax_all, ay_all, ax_front, ay_front, sx, sy, xlabel, ylabel in panels:
        ax.set_facecolor(BG)
        ax.grid(True)

        # All trials (background)
        ax.scatter(
            ax_all, ay_all,
            c=GREY_DOT, s=18, alpha=0.35, linewidths=0,
            label=f"All trials ({len(all_trials)})",
            zorder=2,
        )

        # Pareto front (coloured by RMSE)
        sc = ax.scatter(
            ax_front, ay_front,
            c=front_rmse, cmap=CMAP, norm=norm,
            s=55, alpha=0.9, linewidths=0.4, edgecolors=TEXT_CLR,
            label=f"Pareto front ({len(pareto_trials)})",
            zorder=3,
        )

        # Selected trial (star)
        ax.scatter(
            sx, sy,
            c=SEL_CLR, marker="*", s=280, zorder=5,
            label=f"Selected (trial #{best_trial.number})",
            edgecolors="white", linewidths=0.6,
        )
        ax.annotate(
            f" #{best_trial.number}",
            xy=(sx, sy),
            color=SEL_CLR, fontsize=8, fontweight="bold",
            xytext=(6, 4), textcoords="offset points",
        )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        # Format x-axis param counts as integers with commas
        if "Parameter" in xlabel:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
            )
        if "Parameter" in ylabel:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
            )

        ax.legend(
            fontsize=7.5, framealpha=0.25,
            facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
        )

    # Shared RMSE colorbar on the right
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("RMSE", fontsize=9, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "pareto_front.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    logger.info(f"Pareto front plot saved → '{save_path}'")
    return save_path


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
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # CosineAnnealingLR is well-suited for short tuning windows: it
        # provides a smooth, predictable decay that does not require
        # plateau detection over 20 epochs.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TUNING_EPOCHS, eta_min=lr * 0.01
        )

        # One scaler per trial — GradScaler maintains internal scale state
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
        best_params   — Hyperparameter dict for the selected Pareto-optimal trial.
        train_scaled  — Scaled training DataFrame (fitted preprocessor).
        val_scaled    — Scaled validation DataFrame (transform only).
        preprocessor  — Fitted CMAPSSPreprocessor instance.
        feature_cols  — Canonical feature column list from the fitted preprocessor.
        study         — Completed Optuna study (passed to finalize_model for plot).
        best_trial    — The FrozenTrial selected by select_best_from_pareto().

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
# Final Model Training
# ---------------------------------------------------------------------------

def finalize_model(
    best_params:  Dict[str, Any],
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    preprocessor: CMAPSSPreprocessor,
    feature_cols: List[str],
    study:        optuna.Study,
    best_trial:   optuna.trial.FrozenTrial,
) -> str:
    """
    Retrain the final model with the selected hyperparameters on the full
    training set.  Bundles the ONNX model, fitted scaler, and feature schema
    into a single versioned directory under MODELS_DIR.

    Training regime:
        - AMP (torch.amp): autocast + GradScaler for faster GPU training.
          Master weights remain fp32 throughout; model.float() is called before
          ONNX export as an explicit fp32 flush of any residual bf16 buffers.
        - torch.compile (PyTorch >= 2.0): compiles the model graph with
          mode="reduce-overhead" for kernel fusion and reduced Python overhead.
          The original (uncompiled) module is extracted via _orig_mod before
          ONNX export since torch.onnx.export requires a traceable module.
        - CosineAnnealingLR: smooth epoch-level decay.
        - Early stopping on NASA score with patience=8.
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

    # torch.compile: fuse kernels and reduce Python dispatch overhead.
    # mode="reduce-overhead" is optimal for small models with many repeated
    # forward passes (the 1D-CNN tile is called thousands of times).
    # Wrapped in try/except so the pipeline degrades gracefully on
    # PyTorch < 2.0 or environments where dynamo is unavailable.
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        logger.info("torch.compile: enabled (mode='reduce-overhead').")
    except Exception as exc:
        compiled_model = model
        logger.warning(f"torch.compile unavailable, running eager mode: {exc}")

    # AMP scaler for final training — persists across all 70 epochs.
    scaler = build_scaler(device)

    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, val_df,
        feature_cols=feature_cols,
        batch_size=64,
        seed=42,
    )

    max_epochs = 70
    optimizer  = optim.AdamW(
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
        train_one_epoch(compiled_model, train_loader, optimizer, device, scaler)
        val_rmse, nasa_score = evaluate(compiled_model, val_loader, device)
        scheduler.step()   # epoch-level step: must be called after evaluate()

        logger.info(
            f"Epoch {epoch + 1:03d}/{max_epochs} | "
            f"val_rmse={val_rmse:.4f}  nasa={nasa_score:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if nasa_score < best_val_nasa:
            best_val_nasa     = nasa_score
            epochs_no_improve = 0
            # Pull state from the original module, not the compiled wrapper,
            # to guarantee a plain state dict that can be loaded into an
            # uncompiled model for ONNX export.
            _src = compiled_model._orig_mod if hasattr(compiled_model, "_orig_mod") else compiled_model
            best_model_state  = {k: v.clone() for k, v in _src.state_dict().items()}
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

    # ONNX export must use the original (uncompiled) module in fp32.
    # - _orig_mod: unwraps the torch.compile wrapper if present.
    # - .float(): flushes any residual bf16 buffers (e.g. BN running stats)
    #   that autocast may have left behind, guaranteeing a pure fp32 graph.
    #   CPUExecutionProvider in ONNX Runtime requires fp32 inputs/weights.
    export_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    export_model.float().cpu().eval()

    # 1. PyTorch Native Save (Dual Exporting Strategy)
    pth_save_path = os.path.join(save_dir, "model.pth")
    set_seed(42)  # Enforce strict determinism prior to PyTorch native serialization
    torch.save(export_model.state_dict(), pth_save_path)
    logger.info(f"Exported PyTorch weights to '{pth_save_path}'.")

    # 2. ONNX Export
    model_save_path = os.path.join(save_dir, "model.onnx")
    export_to_onnx(export_model, model_save_path, input_shape=(1, 30, input_channels))
    logger.info(f"Exported ONNX model to '{model_save_path}'.")

    # 3. Hyperparameter Export
    hparams_save_path = os.path.join(save_dir, "hyperparameters.json")
    with open(hparams_save_path, "w") as f:
        import json
        json.dump(best_params, f, indent=4)
    logger.info(f"Exported hyperparameters to '{hparams_save_path}'.")

    preprocessor.save_artifacts(save_dir)

    # Export Pareto front visualisation into the same versioned directory.
    plot_pareto_front(study, best_trial, save_dir)

    logger.info(
        f"Artifact bundle complete → '{save_dir}'\n"
        f"  • model.pth           ({os.path.getsize(pth_save_path) / 1024:.1f} KB)\n"
        f"  • model.onnx          ({os.path.getsize(model_save_path) / 1024:.1f} KB)\n"
        f"  • scaler.joblib\n"
        f"  • feature_schema.json\n"
        f"  • pareto_front.png\n"
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
        best_hparams, train_data, val_data, preprocessor, feature_cols, study, best_trial = result
        finalize_model(best_hparams, train_data, val_data, preprocessor, feature_cols, study, best_trial)