"""src/Visualize.py — 3D genome rendering and material frequency analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from src.milestone1.genome import ACTIVE_N, ACTIVE_P, EMPTY, PASSIVE

pv.OFF_SCREEN = True

MAT_COLORS = {
    PASSIVE:  [0.8, 0.8, 0.8],
    ACTIVE_P: [0.2, 0.6, 1.0],
    ACTIVE_N: [1.0, 0.3, 0.1],
}


def render_genome(genome: np.ndarray, title: str = 'Xenobot', save_path: str = None) -> pv.Plotter:
    """Render voxel genome as coloured cubes using PyVista.

    Args:
        genome:    (8,8,8) int array
        title:     plot title string
        save_path: PNG path to save (None = interactive display)
    Returns:
        pv.Plotter instance
    """
    plotter = pv.Plotter(off_screen=save_path is not None)
    for mat, color in MAT_COLORS.items():
        coords = np.argwhere(genome == mat)
        for x, y, z in coords:
            cube = pv.Cube(center=(x, y, z), x_length=0.95, y_length=0.95, z_length=0.95)
            plotter.add_mesh(cube, color=color, opacity=0.9)
    plotter.add_text(title, font_size=12)
    plotter.show_grid()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
        plotter.screenshot(save_path)
    else:
        plotter.show()
    return plotter


def render_hof(
    hof_genomes: np.ndarray,
    fitnesses: list,
    save_dir: str = 'renders/m3/',
) -> None:
    """Render each genome in HallOfFame and save PNG.

    Args:
        hof_genomes: array of shape (N, 8, 8, 8) or list of (8,8,8) arrays
        fitnesses:   list of fitness scores, same length as hof_genomes
        save_dir:    directory for output PNGs
    """
    os.makedirs(save_dir, exist_ok=True)
    for i, (genome, score) in enumerate(zip(hof_genomes, fitnesses)):
        path = os.path.join(save_dir, f'hof_{i}_fitness_{score:.4f}.png')
        render_genome(np.asarray(genome, dtype=int), title=f'HoF #{i}  fit={score:.4f}', save_path=path)
        print(f'Saved {path}')


def material_frequency_heatmap(
    genomes: list,
    save_path: str = 'results/m3/material_heatmap.png',
) -> None:
    """Plot active-voxel frequency heatmap over top-N genomes.

    For each (x,y,z), computes the fraction of genomes where the voxel is
    active (material 2 or 3). Sums over Z to produce an 8×8 heatmap.

    Args:
        genomes:   list of (8,8,8) numpy arrays
        save_path: output PNG path
    """
    n = len(genomes)
    arr = np.stack([np.asarray(g, dtype=int) for g in genomes], axis=0)  # (N,8,8,8)
    active_frac = ((arr == ACTIVE_P) | (arr == ACTIVE_N)).mean(axis=0)    # (8,8,8)
    heatmap = active_frac.sum(axis=2)                                      # (8,8) sum over Z

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, origin='lower', cmap='hot', vmin=0, vmax=heatmap.max())
    fig.colorbar(im, ax=ax, label='Active voxel frequency (sum over Z)')
    ax.set_xlabel('Y axis')
    ax.set_ylabel('X axis')
    ax.set_title(f'Active voxel frequency — top-{n} robots')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved material heatmap → {save_path}')


# ── M4 additions ──────────────────────────────────────────────────────────────

def render_pareto_trio(
    most_mobile: np.ndarray,
    most_efficient: np.ndarray,
    knee_point: np.ndarray,
    fitnesses: dict,
    save_path: str = 'renders/m4/pareto_trio.png',
) -> None:
    """Render three Pareto-extreme robots as individual PNG files.

    Saves three files alongside save_path:
        renders/m4/pareto_most_mobile.png
        renders/m4/pareto_most_efficient.png
        renders/m4/pareto_knee.png

    Args:
        most_mobile:    genome with highest locomotion distance
        most_efficient: genome with fewest voxels
        knee_point:     knee-point genome on the Pareto front
        fitnesses:      dict with keys 'mobile', 'efficient', 'knee' → (dist, n_voxels)
        save_path:      base path (directory is derived from this)
    """
    pv.OFF_SCREEN = True
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    cases = [
        (most_mobile,    os.path.join(save_dir, 'pareto_most_mobile.png'),
         f'Most Mobile  d={fitnesses.get("mobile", (0,))[0]:.4f}'),
        (most_efficient, os.path.join(save_dir, 'pareto_most_efficient.png'),
         f'Most Efficient  v={fitnesses.get("efficient", (0, 0))[1]:.0f}'),
        (knee_point,     os.path.join(save_dir, 'pareto_knee.png'),
         f'Knee Point'),
    ]
    for genome, path, title in cases:
        if genome is None:
            continue
        render_genome(np.asarray(genome, dtype=int), title=title, save_path=path)
        print(f'Saved {path}')


def render_evolution_stages(
    checkpoints: list,
    save_dir: str = 'renders/m4/stages/',
) -> None:
    """Render genomes at different evolution stages.

    Args:
        checkpoints: list of (genome, fitness, generation) tuples
        save_dir:    output directory
    """
    pv.OFF_SCREEN = True
    os.makedirs(save_dir, exist_ok=True)

    for genome, fitness, generation in checkpoints:
        path = os.path.join(save_dir, f'gen_{generation:04d}_fit_{fitness:.4f}.png')
        render_genome(
            np.asarray(genome, dtype=int),
            title=f'Gen {generation} | fit={fitness:.4f}',
            save_path=path,
        )
        print(f'Saved {path}')
