"""Interactively compare three-objective WFG targets with theoretical fronts.

References: Huband et al. (2006), doi:10.1109/TEVC.2005.861417; the full WFG3
line and sheet follow Ishibuchi et al. (2016), doi:10.1109/TEVC.2015.2505784.
Reference meshes use the equations directly, independently of the target code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from hvd import problems

BLUE, GREEN = "#2166b1", "#29816b"


def wfg2_intervals():
    """Efficient branches are the strict records of u*cos(5*pi*u)**2."""
    record = lambda u: u * np.cos(5 * np.pi * u) ** 2
    intervals, previous_peak = [], 0.0
    for index in range(6):
        center = index / 5
        peak = 1.0 if index == 5 else brentq(
            lambda u: np.tan(5 * np.pi * u) - 1 / (10 * np.pi * u),
            max(center, 1e-12), center + 0.05,
        )
        left = 0.0 if index == 0 else brentq(
            lambda u: record(u) - previous_peak, center - 0.1, center,
        )
        # Later branch starts tie an earlier peak and are excluded.
        intervals.append((left + (peak - left) * 1e-8 if index else left, peak))
        previous_peak = record(peak)
    return intervals


def reference_geometry(number, resolution=65):
    """Return theoretical surface meshes and, for WFG3, its efficient line."""
    if number in (1, 2):
        intervals = [(0.0, 1.0)] if number == 1 else wfg2_intervals()
        meshes = []
        for left, right in intervals:
            u, v = np.meshgrid(np.linspace(left, right, resolution), np.linspace(0, 1, resolution))
            amplitude = 1 - np.cos(np.pi * u / 2)
            f1 = 2 * amplitude * (1 - np.cos(np.pi * v / 2))
            f2 = 4 * amplitude * (1 - np.sin(np.pi * v / 2))
            f3 = 6 * (1 - u + np.sin(10 * np.pi * u) / (10 * np.pi)) if number == 1 else (
                6 * (1 - u * np.cos(5 * np.pi * u) ** 2)
            )
            meshes.append((f1, f2, f3))
        return meshes, None
    if number == 3:
        # Ishibuchi et al., Eq. (25): parameterize the sheet by f3 and distance.
        f3, fraction = np.meshgrid(
            np.linspace(1e-10, 3 - 1e-10, resolution), np.linspace(1e-9, 1, resolution)
        )
        lower, upper = np.maximum(0, f3 - 2), np.minimum(1, f3)
        distance = lower + (upper - lower) * fraction
        position = 1 + (distance - f3) / 6
        sheet = (distance + position * (1 + distance), distance + 2 * position * (1 - distance), f3)
        t = np.linspace(0, 1, 300)
        return [sheet], np.array([t, 2 * t, 6 * (1 - t)])
    theta, phi = np.meshgrid(
        np.linspace(0, np.pi / 2, resolution), np.linspace(0, np.pi / 2, resolution)
    )
    sphere = (2 * np.sin(theta) * np.cos(phi), 4 * np.sin(theta) * np.sin(phi), 6 * np.cos(theta))
    return [sphere], None


def plot_fronts(points=500):
    """Create the comparison figure; GUI backends allow rotating each panel."""
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    for number in range(1, 10):
        ax = fig.add_subplot(3, 3, number, projection="3d")
        meshes, line = reference_geometry(number)
        for mesh in meshes:
            ax.plot_surface(*mesh, color=GREEN, alpha=0.06, linewidth=0, shade=False)
            ax.plot_wireframe(*mesh, rstride=8, cstride=8, color=GREEN, alpha=0.65, linewidth=0.6)
        if line is not None:
            ax.plot(*line, color=GREEN, linewidth=1.5)
        front = getattr(problems, f"WFG{number}")(n_var=10, n_obj=3).get_pareto_front(points)
        ax.scatter(*front.T, s=7, c=BLUE, depthshade=False, edgecolors="none")
        ax.set(
            title=f"WFG{number}", xlabel="f1", ylabel="f2", zlabel="f3",
            xlim=(0, 3.05 if number == 3 else 2.05),
            ylim=(0, 2.05 if number == 3 else 4.1), zlim=(0, 6.1),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=43)
        ax.tick_params(labelsize=8, pad=0)
    fig.suptitle("WFG1–9: target samples and theoretical fronts", fontsize=16)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=BLUE, label=f"Target ({points} points)"),
            Line2D([], [], color=GREEN, label="Theoretical front"),
        ],
        loc="outside lower center", ncol=2, frameon=False,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=500, help="target samples per problem (default: 500)")
    parser.add_argument("--output", type=Path, help="optionally save the figure to an image file")
    parser.add_argument("--no-show", action="store_true", help="skip the interactive window (for saving only)")
    args = parser.parse_args()
    if args.points < 3:
        parser.error("--points must be at least 3")
    fig = plot_fronts(args.points)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
