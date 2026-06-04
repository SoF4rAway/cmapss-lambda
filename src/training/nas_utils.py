import optuna
import numpy as np
import os
import logging
import matplotlib
matplotlib.use("Agg")   # headless — no display required on training server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("RUL_NAS_Utils")

# ---------------------------------------------------------------------------
# Objective priority weights for Pareto-front selection.
# NASA Asymmetric Score is the primary metric (safety-critical correctness).
# Model parameter count is the efficiency proxy (L1/L2 cache residency goal).
# RMSE is a useful secondary signal but not the primary concern.
# ---------------------------------------------------------------------------
PARETO_WEIGHTS = (0.65, 0.25, 0.10)   # (nasa_score, n_params, rmse)


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

    Layout (1 row x 3 columns):
        [0] NASA Score  vs  Parameter Count   — primary accuracy/efficiency plane
        [1] NASA Score  vs  RMSE              — accuracy decomposition
        [2] Parameter Count  vs  RMSE         — efficiency vs auxiliary accuracy

    Visual encoding:
        - All completed trials      — grey circles, alpha=0.35
        - Pareto-front trials       — filled circles, coloured by RMSE (viridis_r)
        - Selected best trial       — red star (*), annotated with trial number

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
        f"NSGA-II Pareto Front  .  {len(pareto_trials)} non-dominated trials"
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

    logger.info(f"Pareto front plot saved -> '{save_path}'")
    return save_path
