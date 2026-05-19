import torch
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
