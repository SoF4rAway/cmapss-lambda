import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import argparse
import json
import optuna
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.models.architecture import RUL_1D_CNN, export_to_onnx
from src.data.loaders import get_dataloaders
from src.data.preprocess import CMAPSSPreprocessor
from src.core.config import set_seed, DATA_DIR, MODELS_DIR
from src.training.utils import build_scaler, train_one_epoch
from src.training.eval import evaluate
from src.training.nas_utils import plot_pareto_front

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RUL_Train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Parameter ceiling above which efficiency pressure kicks in (aligns with
# the <500KB / <1MB model size targets in ARCHITECTURE.md).
PARAM_BUDGET = 50_000


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------

def finalize_model(
    best_params:  Dict[str, Any],
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    preprocessor: CMAPSSPreprocessor,
    feature_cols: List[str],
    study:        Optional[optuna.Study] = None,
    best_trial:   Optional[optuna.trial.FrozenTrial] = None,
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

    Args:
        best_params:   Hyperparameter dict (from tuning or manual CLI).
        train_df:      Scaled training DataFrame.
        val_df:        Scaled validation DataFrame.
        preprocessor:  Fitted CMAPSSPreprocessor (used for artifact saving).
        feature_cols:  Canonical feature column list.
        study:         Optional Optuna study (for Pareto plot when available).
        best_trial:    Optional selected trial (for Pareto plot when available).

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
        json.dump(best_params, f, indent=4)
    logger.info(f"Exported hyperparameters to '{hparams_save_path}'.")

    preprocessor.save_artifacts(save_dir)

    # Export Pareto front visualisation into the same versioned directory (if tuning).
    if study is not None and best_trial is not None:
        plot_pareto_front(study, best_trial, save_dir)

    # Compute artifact sizes for logging (only if files were written).
    pth_size = os.path.getsize(pth_save_path) / 1024
    logger.info(
        f"Artifact bundle complete -> '{save_dir}'\n"
        f"  - model.pth           ({pth_size:.1f} KB)\n"
        f"  - model.onnx          ({os.path.getsize(model_save_path) / 1024:.1f} KB)\n"
        f"  - scaler.joblib\n"
        f"  - feature_schema.json\n"
        f"  - {'pareto_front.png' if study is not None and best_trial is not None else '(skipped - no tuning data)'}\n"
        f"  Best val nasa_score: {best_val_nasa:.2f}"
    )
    return save_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with two subcommand modes."""
    parser = argparse.ArgumentParser(
        description="RUL_1D_CNN final model retraining and ONNX export.",
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # --- Mode: from-tuning ---
    ft_parser = subparsers.add_parser(
        "from-tuning",
        help="Load hyperparameters from a JSON file produced by a prior tuning run.",
    )
    ft_parser.add_argument(
        "--hparams-file", type=str, required=True,
        help="Path to hyperparameters.json from a tuning run.",
    )
    _add_shared_args(ft_parser)

    # --- Mode: manual ---
    manual_parser = subparsers.add_parser(
        "manual",
        help="Specify all architecture and training parameters directly.",
    )
    _add_shared_args(manual_parser)
    _add_architecture_args(manual_parser)

    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both modes."""
    parser.add_argument("--max-rul", type=int, default=125, help="RUL clip ceiling.")
    parser.add_argument("--max-epochs", type=int, default=70, help="Maximum training epochs.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on NASA score.")
    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of engines held for validation.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing CMAPSS .txt files (overrides DATA_DIR).")
    parser.add_argument("--train-file", type=str, default="train_FD001.txt", help="Training data filename.")
    parser.add_argument("--output-dir", type=str, default=None, help="Artifact output directory override.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")


def _add_architecture_args(parser: argparse.ArgumentParser) -> None:
    """Add architecture hyperparameter arguments for manual mode."""
    parser.add_argument("--num-blocks", type=int, required=True, choices=[1, 2, 3, 4], help="Number of depthwise-separable conv blocks.")
    parser.add_argument("--kernel-size", type=int, required=True, choices=[3, 5, 7], help="Conv kernel size.")
    parser.add_argument("--dilation", type=int, required=True, choices=[1, 2, 3], help="Conv dilation factor.")
    parser.add_argument("--fc-units", type=int, required=True, choices=[16, 32, 64], help="FC layer width.")
    parser.add_argument("--out-channels", type=int, nargs="+", required=True, help="Space-separated per-block output channels.")
    parser.add_argument("--use-bn", type=lambda x: x.lower() == "true", nargs="+", required=True, help="Space-separated per-block BN flags (true/false).")
    parser.add_argument("--dropouts", type=float, nargs="+", required=True, help="Space-separated per-block dropout rates.")
    parser.add_argument("--lr", type=float, required=True, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, required=True, help="AdamW weight decay.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for standalone final model retraining."""
    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.error("Choose a mode: 'from-tuning' or 'manual'.")

    set_seed(args.seed)

    # --- Resolve data directory ---
    cmapss_data_dir = os.path.join(args.data_dir, "CMAPSSData") if args.data_dir else os.path.join(DATA_DIR, "CMAPSSData")
    train_path = os.path.join(cmapss_data_dir, args.train_file)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    # --- Hyperparameters ---
    if args.mode == "from-tuning":
        with open(args.hparams_file, "r") as f:
            best_params: Dict[str, Any] = json.load(f)
        logger.info(f"Loaded hyperparameters from '{args.hparams_file}'.")
    else:
        # Manual mode: validate list lengths match num_blocks
        if len(args.out_channels) != args.num_blocks:
            parser.error(f"--out-channels requires {args.num_blocks} values, got {len(args.out_channels)}.")
        if len(args.use_bn) != args.num_blocks:
            parser.error(f"--use-bn requires {args.num_blocks} values, got {len(args.use_bn)}.")
        if len(args.dropouts) != args.num_blocks:
            parser.error(f"--dropouts requires {args.num_blocks} values, got {len(args.dropouts)}.")

        best_params = {
            "num_blocks": args.num_blocks,
            "kernel_size": args.kernel_size,
            "dilation": args.dilation,
            "fc_units": args.fc_units,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        for i in range(args.num_blocks):
            best_params[f"out_channels_b{i}"] = args.out_channels[i]
            best_params[f"use_bn_b{i}"]       = args.use_bn[i]
            best_params[f"dropout_b{i}"]      = args.dropouts[i]

        logger.info(f"Built hyperparameter dict from manual CLI: {best_params}")

    # --- Preprocessing (re-fit from scratch, zero temporal leakage) ---
    preprocessor = CMAPSSPreprocessor(max_rul=args.max_rul)
    raw_train    = preprocessor.load_data(train_path)
    raw_train    = preprocessor.add_piecewise_rul(raw_train)
    train_set, val_set = preprocessor.split_train_val_by_engine(
        raw_train, val_ratio=args.val_ratio, seed=args.seed,
    )

    # Scaler fitted strictly on training data (zero temporal leakage).
    train_scaled = preprocessor.fit_transform(train_set)
    val_scaled   = preprocessor.transform(val_set)

    feature_cols: List[str] = preprocessor.active_features
    logger.info(f"Feature schema locked: {len(feature_cols)} active features.")

    # --- Run final training ---
    finalize_model(
        best_params=best_params,
        train_df=train_scaled,
        val_df=val_scaled,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
    )


if __name__ == "__main__":
    main()
