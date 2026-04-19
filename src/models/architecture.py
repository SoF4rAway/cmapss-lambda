import os
import torch
import torch.nn as nn
import torch.onnx
from typing import List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel recalibration.

    Uses Conv1d(1×1) as a reshape-free substitute for nn.Linear, allowing
    the block to sit inline within a (B, C, T) feature pipeline.

    Placement: applied AFTER BN+GELU so the squeeze statistics reflect
    activated channel magnitudes, not raw pre-activation values.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, reduced,     kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(reduced,     in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.fc(self.avg_pool(x))   # (B, C, 1)
        return x * y


class TemporalAttentionPool(nn.Module):
    """
    Learned temporal attention pooling.

    Replaces AdaptiveAvgPool1d(1), which assigns equal weight to every
    timestep.  For RUL prediction the most recent cycles carry the heaviest
    degradation signal; this module learns to weight them accordingly via a
    single trainable Conv1d(C → 1, kernel=1).

    Cost: exactly C extra parameters — negligible.
    ONNX-compatible at opset 14 (softmax + einsum-equivalent matmul).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.scorer = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        w = torch.softmax(self.scorer(x), dim=-1)  # (B, 1, T)
        return (x * w).sum(dim=-1)                  # (B, C)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D-Conv block with SE attention and residual connection.

    Operation order:
        Depthwise → Pointwise → BN → GELU → SE → Dropout

    Why this order?
    ---------------
    The original code ordered SE before BN+GELU.  That is wrong for two reasons:
      1. Pre-Norm (BN before activation) is the standard convention for 1D conv
         blocks and stabilises training.
      2. SE channel attention should see which channels are *firing* after the
         non-linearity.  Placing SE after GELU produces more informative squeeze
         statistics and better channel recalibration.

    Residual connection:
        A 1×1 projection conv is inserted when in_channels ≠ out_channels.
        This is critical for stable gradient flow in num_blocks=3–4 NAS
        configurations where the sequential stack can otherwise suffer from
        vanishing gradients.
        The same-length output guarantee:
            padding = (dilation × (kernel_size − 1)) // 2
        ensures the spatial dimension is preserved for all (kernel_size, dilation)
        combinations in the NAS search space, making residual addition safe.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        use_bn:       bool,
        dropout:      float,
        dilation:     int,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2

        # Depthwise: per-channel temporal filtering
        self.depthwise = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
            dilation=dilation,
        )
        # Pointwise: cross-channel mixing
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        # Pre-Norm → activation → SE → dropout
        self.bn      = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.gelu    = nn.GELU()
        self.se      = SEBlock(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Residual projection: identity when dims match, 1×1 conv otherwise.
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.se(out)
        out = self.dropout(out)
        return out + residual


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class RUL_1D_CNN(nn.Module):
    """
    MobileNet-inspired 1D-CNN for Remaining Useful Life (RUL) regression.

    Key design decisions
    --------------------
    - Depthwise Separable Convolutions reduce parameter count while preserving
      representational capacity (aligned with <500KB model size target).
    - SE channel attention recalibrates feature importance per block.
    - Residual connections ensure gradient health for deeper NAS configs.
    - TemporalAttentionPool replaces naive average pooling so the model can
      learn to weight recent degradation cycles more heavily.
    - Per-block configuration of channels, BN, and dropout enables NSGA-II
      NAS without any architecture-side changes.

    Input convention (matches DataLoader output):
        x: (Batch, SeqLen, Features)  — note: NOT (B, C, T).
        The forward() method transposes internally before conv layers.
    """

    def __init__(
        self,
        input_channels:    int        = 14,     # FD001: ~14 active features post-pruning
        num_blocks:        int        = 2,
        out_channels_list: List[int]  = None,
        kernel_size:       int        = 3,
        use_bn_list:       List[bool] = None,
        dropout_list:      List[float]= None,
        fc_units:          int        = 32,
        dilation:          int        = 1,
    ):
        super().__init__()

        # Default list arguments (avoid mutable defaults)
        if out_channels_list is None:
            out_channels_list = [32] * num_blocks
        if use_bn_list is None:
            use_bn_list = [True] * num_blocks
        if dropout_list is None:
            dropout_list = [0.1] * num_blocks

        # Strict length validation — silent tiling would silently mask NAS
        # parameter mismatches and corrupt the search space.
        if len(out_channels_list) != num_blocks:
            raise ValueError(
                f"out_channels_list has {len(out_channels_list)} entries "
                f"but num_blocks={num_blocks}."
            )
        if len(use_bn_list) != num_blocks:
            raise ValueError(
                f"use_bn_list has {len(use_bn_list)} entries "
                f"but num_blocks={num_blocks}."
            )
        if len(dropout_list) != num_blocks:
            raise ValueError(
                f"dropout_list has {len(dropout_list)} entries "
                f"but num_blocks={num_blocks}."
            )

        blocks = []
        in_c = input_channels
        for i in range(num_blocks):
            out_c = out_channels_list[i]
            blocks.append(
                DepthwiseSeparableConv1d(
                    in_c, out_c,
                    kernel_size=kernel_size,
                    use_bn=use_bn_list[i],
                    dropout=dropout_list[i],
                    dilation=dilation,
                )
            )
            in_c = out_c

        self.feature_extractor = nn.Sequential(*blocks)

        # Temporal attention pool: learned timestep weighting → (B, C)
        # Replaces AdaptiveAvgPool1d(1) + Flatten.
        self.temporal_pool = TemporalAttentionPool(in_c)

        self.fc = nn.Sequential(
            nn.Linear(in_c, fc_units),
            nn.GELU(),
            nn.Dropout(dropout_list[-1]) if dropout_list[-1] > 0.0 else nn.Identity(),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, SeqLen, Features)
        """
        x = x.transpose(1, 2)            # → (Batch, Features, SeqLen) for Conv1d
        x = self.feature_extractor(x)    # → (Batch, out_channels[-1], SeqLen)
        x = self.temporal_pool(x)        # → (Batch, out_channels[-1])
        x = self.fc(x)                   # → (Batch, 1)
        return x


# ---------------------------------------------------------------------------
# ONNX Export & Validation
# ---------------------------------------------------------------------------

def export_to_onnx(
    model:       nn.Module,
    save_path:   str,
    input_shape: Tuple[int, int, int] = (1, 30, 14),
) -> int:
    """
    Export model to ONNX (opset 14) with a dynamic batch axis.

    Args:
        model:       Trained RUL_1D_CNN in eval mode.
        save_path:   Destination .onnx file path.
        input_shape: (batch, seq_len, features) matching forward() convention.

    Returns:
        File size in bytes (for caller logging / size-budget assertions).
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    file_bytes = os.path.getsize(save_path)
    print(f"Model exported → {save_path}  ({file_bytes / 1024:.1f} KB)")
    return file_bytes


def validate_onnx(
    onnx_path:   str,
    dummy_input: np.ndarray,
    max_rul:     float = 125.0,
) -> np.ndarray:
    """
    Validate the exported ONNX model with shape, numerical, and range checks.

    Checks performed:
        1. Output shape is (N, 1).
        2. No NaN or Inf values.
        3. Predictions within a loose physical range [−max_rul, 2×max_rul].
           Out-of-range predictions are logged as warnings (not hard failures)
           since they may legitimately occur early in training.

    Args:
        onnx_path:   Path to the exported .onnx file.
        dummy_input: Input array of shape (N, seq_len, features), dtype float32.
        max_rul:     RUL cap used during preprocessing (default 125, per ARCHITECTURE.md).

    Returns:
        predictions: np.ndarray of shape (N, 1).
    """
    import onnxruntime as ort

    session     = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preds = session.run([output_name], {input_name: dummy_input.astype(np.float32)})[0]

    # --- Shape check ---
    if preds.ndim != 2 or preds.shape[1] != 1:
        raise ValueError(f"Expected output shape (N, 1), got {preds.shape}.")

    # --- Numerical sanity ---
    if np.isnan(preds).any():
        raise ValueError("ONNX output contains NaN values.")
    if np.isinf(preds).any():
        raise ValueError("ONNX output contains Inf values.")

    # --- Loose physical range warning ---
    lo, hi = -max_rul, 2.0 * max_rul
    n_out  = int(np.sum((preds < lo) | (preds > hi)))
    if n_out > 0:
        print(
            f"[validate_onnx] WARNING: {n_out}/{len(preds)} predictions outside "
            f"loose physical range [{lo:.0f}, {hi:.0f}].  "
            f"min={preds.min():.2f}  max={preds.max():.2f}"
        )

    print(
        f"[validate_onnx] OK — shape={preds.shape}  "
        f"min={preds.min():.2f}  max={preds.max():.2f}  "
        f"mean={preds.mean():.2f}"
    )
    return preds