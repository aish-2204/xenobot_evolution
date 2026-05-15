"""src/milestone3/visualize_m3.py

Diagnostic plots for Milestone 3 ablation study.

Functions
---------
plot_ablation_bars(results, save_path)      — bar chart, error bars, p-values
plot_ablation_curves(results, save_path)    — fitness curves per condition
plot_fitness_heatmap_m3(hof, save_path)     — voxel active-frequency heatmap
render_best_per_condition(results, save_dir)— PyVista 3D renders per condition
"""

import csv
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.milestone1.visualize import render_genome
from src.milestone3.ablation import ABLATION_CONDITIONS


# ── Colour palette (consistent across all M3 plots) ─────────────────────────

_COLOURS = {
    "baseline":         "#2196F3",
    "no_crossover":     "#FF9800",
    "no_connectivity":  "#9C27B0",
    "random_selection": "#F44336",
    "single_material":  "#795548",
    "no_mutation":      "#009688",
}


# ── Plot 1: Ablation bar chart ───────────────────────────────────────────────

def plot_ablation_bars(
    results: Dict[str, List],
    save_path: str,
    stats_csv: Optional[str] = None,
) -> None:
    """Bar chart of best-fitness per condition with error bars and p-values.

    Args:
        results:   {condition: [(logbook, hof), ...]} from run_ablation()
        save_path: output PNG path
        stats_csv: optional path to ablation_stats.csv (for p-values)
    """
    conditions = [c for c in ABLATION_CONDITIONS if c in results]
    means, stds, pvals = [], [], {}

    # Load p-values if stats CSV exists
    if stats_csv and os.path.exists(stats_csv):
        with open(stats_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pvals[row["condition"]] = row["mannwhitney_p"]

    for cond in conditions:
        runs  = results[cond]
        bests = [max(lb.select("max")) for lb, _ in runs]
        means.append(float(np.mean(bests)))
        stds.append(float(np.std(bests)))

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(conditions))
    colors = [_COLOURS.get(c, "gray") for c in conditions]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", width=0.6)

    # Annotate p-values above bars (skip baseline)
    for i, cond in enumerate(conditions):
        if cond == "baseline":
            ax.text(i, means[i] + stds[i] + 0.002, "reference",
                    ha="center", va="bottom", fontsize=7, color="gray")
        elif cond in pvals:
            p = pvals[cond]
            try:
                p_f = float(p)
                label = "***" if p_f < 0.001 else ("**" if p_f < 0.01 else ("*" if p_f < 0.05 else f"p={p_f:.2f}"))
            except (ValueError, TypeError):
                label = f"p={p}"
            ax.text(i, means[i] + stds[i] + 0.002, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=9)
    ax.set_ylabel("Best fitness (NormFinalDist)")
    ax.set_title("Ablation study — best fitness per condition\n(error bars = std across seeds)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.25 + 0.01)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    _write_csv(
        os.path.splitext(save_path)[0] + ".csv",
        ["condition", "mean_best", "std_best"],
        zip(conditions, means, stds),
    )
    print(f"Saved {save_path}")


# ── Plot 2: Fitness curves per condition ─────────────────────────────────────

def plot_ablation_curves(
    results: Dict[str, List],
    save_path: str,
) -> None:
    """Fitness curves for every condition (mean ± min/max band across seeds).

    Args:
        results:   {condition: [(logbook, hof), ...]} from run_ablation()
        save_path: output PNG path
    """
    conditions = [c for c in ABLATION_CONDITIONS if c in results]

    fig, ax = plt.subplots(figsize=(12, 5))

    for cond in conditions:
        runs   = results[cond]
        gen    = np.array(runs[0][0].select("gen"))
        curves = np.array([np.array(lb.select("max")) for lb, _ in runs])
        mean_c = curves.mean(axis=0)
        min_c  = curves.min(axis=0)
        max_c  = curves.max(axis=0)
        color  = _COLOURS.get(cond, "gray")

        ax.plot(gen, mean_c, color=color, linewidth=2,
                label=cond.replace("_", " "))
        ax.fill_between(gen, min_c, max_c, alpha=0.15, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness (NormFinalDist)")
    ax.set_title("Ablation study — fitness curves\n(shaded band = min–max across seeds)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Plot 3: Voxel heatmap (reused from M2 diagnostics) ──────────────────────

def plot_fitness_heatmap_m3(hof, save_path: str) -> None:
    """Voxel active-frequency heatmap for M3 hall-of-fame.

    Shows active-material frequency per cell across all HOF individuals.

    Args:
        hof:       DEAP HallOfFame
        save_path: output PNG path
    """
    from src.diagnostics import plot_voxel_heatmap
    plot_voxel_heatmap(hof, save_path)


# ── Plot 4: 3D renders of best robot per condition ───────────────────────────

def render_best_per_condition(
    results: Dict[str, List],
    save_dir: str,
) -> Dict[str, str]:
    """Render the best genome from each condition using PyVista.

    Args:
        results:  {condition: [(logbook, hof), ...]} from run_ablation()
        save_dir: directory to save PNG files

    Returns:
        dict {condition: png_path}
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = {}

    for cond, runs in results.items():
        # Find globally best individual across all seeds
        best_ind  = None
        best_fit  = -1.0
        for _, hof in runs:
            for ind in hof:
                f = ind.fitness.values[0]
                if f > best_fit:
                    best_fit = f
                    best_ind = ind

        if best_ind is None:
            continue

        path = os.path.join(save_dir, f"best_{cond}.png")
        try:
            render_genome(np.asarray(best_ind, dtype=int), save_path=path)
            paths[cond] = path
            print(f"Saved render: {path}  (fitness={best_fit:.4f})")
        except Exception as e:
            print(f"Render failed for {cond}: {e}")

    return paths


# ── Summary figure: grid of best renders ────────────────────────────────────

def plot_render_grid(render_paths: Dict[str, str], save_path: str) -> None:
    """Arrange per-condition PyVista renders in a matplotlib grid.

    Args:
        render_paths: {condition: png_path}  from render_best_per_condition()
        save_path:    output PNG path
    """
    conditions = [c for c in ABLATION_CONDITIONS if c in render_paths]
    n = len(conditions)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(nrows, ncols)

    fig.suptitle("Best genome per ablation condition", fontsize=13)

    for idx, cond in enumerate(conditions):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        path = render_paths[cond]
        if os.path.exists(path):
            img = plt.imread(path)
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(cond.replace("_", "\n"), fontsize=9)

    # Hide unused axes
    for idx in range(len(conditions), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Helper ───────────────────────────────────────────────────────────────────

def _write_csv(path: str, header: list, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
