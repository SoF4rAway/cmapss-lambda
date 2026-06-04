import torch
import torch.nn as nn
import numpy as np

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

def asymmetric_loss(y_pred: torch.Tensor, y_true: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Stabilized NASA Asymmetric Loss for PyTorch.
    Supports 'mean', 'sum', and 'none' reductions.
    """
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    d = y_pred - y_true
    d_clamped = torch.clamp(d, min=-65, max=50) 
    loss = torch.where(
        d_clamped < 0, 
        torch.exp(-d_clamped / 15.0) - 1.0, 
        torch.exp(d_clamped / 8.0) - 1.0
    )
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return torch.sum(loss)
    return torch.mean(loss)


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
    Clipping scaled gradients would produce norms ~1000x too large and silently
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
