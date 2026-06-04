import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Tuple

from src.training.utils import asymmetric_loss, nasa_asymmetric_score

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("RUL_Eval")


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
