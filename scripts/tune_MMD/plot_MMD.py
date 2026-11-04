"""Front inspection plots and distribution diagnostics for MMD benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from hvd.utils import get_non_dominated


def distribution_metrics(y, front):
    y = np.asarray(y)
    unique = np.unique(y, axis=0)
    distances = cdist(unique, unique)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1) if len(unique) > 1 else np.array([0.0])
    coverage = cdist(front, unique).min(axis=1)
    convergence = cdist(unique, front).min(axis=1)
    gd = float(np.sqrt(np.mean(convergence**2)))
    igd = float(np.sqrt(np.mean(coverage**2)))
    scale = np.ptp(front, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    normalized_distances = cdist(unique / scale, unique / scale)
    np.fill_diagonal(normalized_distances, np.inf)
    normalized_nearest = normalized_distances.min(axis=1) if len(unique) > 1 else np.array([0.0])
    return {
        "n_points": len(y),
        "n_nondominated": len(get_non_dominated(y)),
        "unique_fraction": float(len(unique) / len(y)),
        "nearest_neighbor_cv": float(nearest.std() / max(nearest.mean(), 1e-30)),
        "normalized_nearest_neighbor_cv": float(
            normalized_nearest.std() / max(normalized_nearest.mean(), 1e-30)
        ),
        "all_point_AHD": max(gd, igd),
        "all_point_GD": gd,
        "all_point_IGD": igd,
        "coverage_p95": float(np.quantile(coverage, 0.95)),
        "coverage_max": float(coverage.max()),
    }


def plot_front(y0, y, reference, front, history, path, title):
    """Plot every final point; filtering only affects metric/nondominated markers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_obj = y.shape[1]
    nd = get_non_dominated(y)
    initial_metrics, final_metrics = distribution_metrics(y0, front), distribution_metrics(y, front)
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    for slot, population, label in ((1, y0, "Initial population"), (2, y, "Final population")):
        ax = fig.add_subplot(2, 3, slot, projection="3d" if n_obj == 3 else None)
        if n_obj == 3:
            ax.scatter(*front.T, s=5, c="#479663", alpha=0.35, label="Target front")
            if slot == 1:
                ax.scatter(*reference.T, s=5, c="#4a80bd", alpha=0.2, label="Reference")
            ax.scatter(*population.T, s=12, c="#343a40" if slot == 1 else "#de8235", label="All points")
            if slot == 2:
                ax.scatter(*nd.T, s=18, marker="+", c="#b52b2b", label="Nondominated")
            ax.set(xlabel="f1", ylabel="f2", zlabel="f3")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(25, 45)
        elif n_obj == 2:
            ax.scatter(*front.T, s=5, alpha=0.3, label="Target front")
            if slot == 1:
                ax.scatter(*reference.T, s=5, alpha=0.2, label="Initial reference")
            ax.scatter(*population.T, s=12, label="All points")
            ax.set(xlabel="f1", ylabel="f2")
        else:
            ax.plot(population.T, alpha=0.2)
            ax.set(xlabel="Objective", ylabel="Value")
        metrics = initial_metrics if slot == 1 else final_metrics
        ax.set_title(
            f"{label}\nAll-point AHD {metrics['all_point_AHD']:.3g}; "
            f"spacing CV {metrics['normalized_nearest_neighbor_cv']:.2f}\n"
            f"Unique points {metrics['unique_fraction']:.1%}",
            fontsize=11,
        )
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 3)
    for key in ("AHD", "GD", "IGD"):
        if key in history:
            values = history[key]
            ax.semilogy(range(len(values)), np.maximum(values, 1e-16), label=key)
    ax.set(xlabel="Newton iteration", ylabel="Distance", title="Convergence (including initialization)")
    ax.set_title("Convergence (including initialization)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend()
    for slot, (i, j) in enumerate(((0, 1), (0, min(2, n_obj - 1)), (1, min(2, n_obj - 1))), 4):
        ax = fig.add_subplot(2, 3, slot)
        ax.scatter(front[:, i], front[:, j], s=5, c="#479663", alpha=0.35, label="Target front")
        ax.scatter(reference[:, i], reference[:, j], s=5, c="#4a80bd", alpha=0.15, label="Initial reference")
        ax.scatter(y0[:, i], y0[:, j], s=8, c="#343a40", alpha=0.25, label="Initial approximation")
        ax.scatter(y[:, i], y[:, j], s=13, c="#de8235", label="All final points")
        ax.scatter(nd[:, i], nd[:, j], s=20, marker="+", c="#b52b2b", label="Nondominated")
        ax.set(xlabel=f"f{i+1}", ylabel=f"f{j+1}", title="Final population projection")
        ax.set_title("Initial / final population projection", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=15)
    fig.savefig(path.with_suffix(".png"), dpi=140)
    plt.close(fig)
