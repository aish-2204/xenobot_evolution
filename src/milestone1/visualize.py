"""3D genome renders (PyVista) and connectivity statistics (Matplotlib)."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from src.milestone1.genome import largest_connected_component, random_genome

_COLORS = {1: "lightgray", 2: "tomato", 3: "cornflowerblue"}
_LABELS = {1: "Passive", 2: "Active+", 3: "Active−"}


def render_genome(genome: np.ndarray, save_path: str | None = None) -> None:
    """3D voxel plot of a genome.

    Colors: passive=gray, active+=red, active-=blue, empty=skipped.
    Shows interactively when save_path is None, otherwise saves a PNG.

    Args:
        genome:    (8,8,8) int array, values 0–3
        save_path: destination PNG path, or None for interactive display
    """
    off_screen = save_path is not None
    pl = pv.Plotter(off_screen=off_screen)

    sx, sy, sz = genome.shape
    grid = pv.ImageData(dimensions=(sx + 1, sy + 1, sz + 1))
    # F-order: x varies fastest, matching PyVista's cell indexing
    grid.cell_data["material"] = genome.flatten(order="F").astype(float)

    for mat_id, color in _COLORS.items():
        sub = grid.threshold([mat_id - 0.5, mat_id + 0.5], scalars="material")
        if sub.n_cells > 0:
            pl.add_mesh(
                sub,
                color=color,
                label=_LABELS[mat_id],
                show_edges=True,
                edge_color="white",
                opacity=0.9,
            )

    pl.add_axes()
    pl.add_legend()
    pl.camera_position = "iso"

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        pl.screenshot(save_path)
        pl.close()
    else:
        pl.show()


def render_connectivity_stats(n_samples: int = 500) -> None:
    """Bar chart of LCC size distribution across random genomes.

    Saves results/m1/connectivity.png and results/m1/connectivity.csv.

    Args:
        n_samples: number of random genomes to sample (seed=42)
    """
    rng = np.random.default_rng(42)
    lcc_sizes = np.array(
        [int(np.sum(largest_connected_component(random_genome(rng)) > 0))
         for _ in range(n_samples)]
    )

    bin_width = 16
    bins = np.arange(0, 513 + bin_width, bin_width)
    counts, edges = np.histogram(lcc_sizes, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(edges[:-1], counts, width=bin_width, align="edge",
           color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(lcc_sizes.mean(), color="crimson", linestyle="--",
               label=f"mean = {lcc_sizes.mean():.1f}")
    ax.set_xlabel("LCC size (non-empty voxels)")
    ax.set_ylabel("Count")
    ax.set_title(f"LCC size distribution — {n_samples} random 8×8×8 genomes")
    ax.legend()
    fig.tight_layout()

    out_dir = "results/m1"
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "connectivity.png")
    csv_path = os.path.join(out_dir, "connectivity.csv")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start", "bin_end", "count"])
        for i, c in enumerate(counts):
            writer.writerow([int(edges[i]), int(edges[i + 1]), int(c)])

    print(f"Saved {png_path} and {csv_path}")
    print(f"LCC sizes — min: {lcc_sizes.min()}, max: {lcc_sizes.max()}, "
          f"mean: {lcc_sizes.mean():.1f}, median: {int(np.median(lcc_sizes))}")


if __name__ == "__main__":
    render_connectivity_stats()
