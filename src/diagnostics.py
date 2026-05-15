"""Diagnostic plots for the Xenobot EA (Milestone 2).

All functions save a PNG and a companion CSV unless noted otherwise.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.milestone1.fitness import evaluate_genome
from src.milestone1.genome import largest_connected_component, random_genome


# ── Plot 1: Fitness curve ───────────────────────────────────────────────────

def plot_fitness_curve(logbook, save_path: str) -> None:
    """Line plot of max and mean fitness with ±1 std shaded band.

    Args:
        logbook:   DEAP Logbook with 'gen', 'max', 'mean', 'std' columns
        save_path: destination PNG path
    """
    gen  = np.array(logbook.select("gen"))
    mx   = np.array(logbook.select("max"))
    mn   = np.array(logbook.select("mean"))
    sd   = np.array(logbook.select("std"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gen, mx, "b-",  linewidth=2, label="Max fitness")
    ax.plot(gen, mn, "g--", linewidth=2, label="Mean fitness")
    ax.fill_between(gen, mn - sd, mn + sd, alpha=0.25, color="green", label="±1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (NormFinalDist)")
    ax.set_title("Fitness over generations")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    _write_csv(os.path.splitext(save_path)[0] + ".csv",
               ["gen", "max", "mean", "std"],
               zip(gen, mx, mn, sd))
    print(f"Saved {save_path}")


# ── Plot 2: Diversity curve ─────────────────────────────────────────────────

def plot_diversity_curve(logbook, save_path: str) -> int | None:
    """Plot mean pairwise Hamming diversity; mark when it falls below 10%.

    Args:
        logbook:   DEAP Logbook with 'gen' and 'diversity' columns
        save_path: destination PNG path

    Returns:
        Generation number where diversity drops below 10% of initial value,
        or None if it never does.
    """
    gen = np.array(logbook.select("gen"))
    div = np.array(logbook.select("diversity"))

    initial   = div[0] if div[0] > 0 else 1e-9
    threshold = 0.1 * initial
    collapse_gen = next((int(g) for g, d in zip(gen, div) if d < threshold), None)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gen, div, color="purple", linewidth=2, label="Diversity (Hamming)")
    ax.axhline(threshold, color="crimson", linestyle="--",
               label=f"10% threshold ({threshold:.4f})")
    if collapse_gen is not None:
        ax.axvline(collapse_gen, color="orange", linestyle=":",
                   label=f"Collapse gen {collapse_gen}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean pairwise Hamming distance")
    ax.set_title("Population diversity over generations")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    _write_csv(os.path.splitext(save_path)[0] + ".csv",
               ["gen", "diversity"], zip(gen, div))
    print(f"Saved {save_path}  |  diversity collapse gen: {collapse_gen}")
    return collapse_gen


# ── Plot 3: Fitness landscape sample ───────────────────────────────────────

def plot_fitness_landscape(
    n_samples: int = 200,
    save_path: str | None = None,
    sim_time: float = 0.5,
    n_workers: int = 4,
) -> np.ndarray:
    """Evaluate n_samples random genomes; plot fitness histogram.

    Args:
        n_samples:  number of genomes to evaluate
        save_path:  destination PNG (or None to just show)
        sim_time:   voxelyze sim duration per call
        n_workers:  parallel threads

    Returns:
        numpy array of fitness scores
    """
    from multiprocessing.pool import ThreadPool

    rng = np.random.default_rng(99)
    genomes = [largest_connected_component(random_genome(rng)) for _ in range(n_samples)]

    def _eval_one(g):
        return evaluate_genome(g, sim_time=sim_time)

    print(f"Evaluating {n_samples} random genomes ({n_workers} threads)…")
    with ThreadPool(n_workers) as pool:
        scores = np.array(pool.map(_eval_one, genomes))

    frac_pos = (scores > 0).mean()
    pos_scores = scores[scores > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: all scores (including zeros)
    ax = axes[0]
    ax.hist(scores, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Fitness")
    ax.set_ylabel("Count")
    ax.set_title(f"All {n_samples} genomes  |  {frac_pos:.1%} score > 0")

    # Right: only positive scores
    ax2 = axes[1]
    if len(pos_scores) > 0:
        ax2.hist(pos_scores, bins=30, color="seagreen", edgecolor="white")
        ax2.axvline(pos_scores.mean(), color="crimson", linestyle="--",
                    label=f"mean = {pos_scores.mean():.3f}")
        ax2.legend()
    ax2.set_xlabel("Fitness (> 0 only)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Positive-fitness genomes  (n={len(pos_scores)})")

    fig.suptitle("Fitness landscape: random genome sample")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        _write_csv(os.path.splitext(save_path)[0] + ".csv",
                   ["score"], ([s] for s in scores))
        print(f"Saved {save_path}")
    plt.close(fig)

    print(f"Fraction with fitness > 0 : {frac_pos:.1%}")
    print(f"Mean (positive only)       : {pos_scores.mean():.4f}" if len(pos_scores) else "No positive scores")
    return scores


# ── Plot 4: Voxel material heatmap ──────────────────────────────────────────

def plot_voxel_heatmap(hof, save_path: str) -> None:
    """Heatmap of material-type frequency per grid cell across HOF individuals.

    Shows: (top row) probability a cell is active (mat 2 or 3),
           (bottom row) probability a cell is occupied (any non-zero).
    Four Z-slices displayed per row.

    Args:
        hof:       DEAP HallOfFame
        save_path: destination PNG
    """
    genomes = np.stack([np.asarray(ind, dtype=int) for ind in hof])  # (n, 8, 8, 8)
    n = len(genomes)

    is_active   = (genomes >= 2).mean(axis=0)   # (8, 8, 8)
    is_occupied = (genomes  > 0).mean(axis=0)

    z_slices = [0, 2, 4, 6]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"Material distribution across top {n} HOF individuals", fontsize=13)

    for col, z in enumerate(z_slices):
        for row, (data, cmap, title_prefix) in enumerate([
            (is_active,   "Reds",  "Active freq"),
            (is_occupied, "Blues", "Occupied freq"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data[:, :, z].T, vmin=0, vmax=1,
                           cmap=cmap, origin="lower", aspect="equal")
            ax.set_title(f"{title_prefix}\nz={z}", fontsize=9)
            ax.set_xlabel("X"); ax.set_ylabel("Y")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Operator comparison plot ─────────────────────────────────────────────────

def plot_operator_comparison(results: dict, save_path: str) -> None:
    """Plot fitness curves for multiple operators with 95% confidence band.

    Args:
        results:   {op_name: [logbook1, logbook2, logbook3]}
        save_path: destination PNG
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"random_flip": "steelblue", "block": "seagreen", "grow_shrink": "tomato"}

    for op, logs in results.items():
        gen = np.array(logs[0].select("gen"))
        # Stack max-fitness curves from all seeds
        curves = np.array([np.array(lb.select("max")) for lb in logs])
        mean_c = curves.mean(axis=0)
        min_c  = curves.min(axis=0)
        max_c  = curves.max(axis=0)
        color  = colors.get(op, "gray")
        ax.plot(gen, mean_c, color=color, linewidth=2, label=op)
        ax.fill_between(gen, min_c, max_c, alpha=0.2, color=color)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title("Operator comparison (3 seeds, shaded = min–max band)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Helper ───────────────────────────────────────────────────────────────────

def _write_csv(path: str, header: list, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
