"""src/milestone4/map_elites.py — MAP-Elites illumination algorithm for xenobots.

Behaviour space: 2D grid (10×10)
  Dimension 1: locomotion distance, 10 bins over [0, 1]
  Dimension 2: bilateral symmetry,  10 bins over [0, 1]

Reference: Mouret & Clune (2015) https://arxiv.org/abs/1504.04909
"""

import json
import os
from multiprocessing import Pool

import numpy as np

from src.ea import mutate_random_flip
from src.milestone1.fitness import evaluate_genome
from src.milestone1.genome import largest_connected_component, random_genome
from src.milestone4.morphological_analysis import bilateral_symmetry_score

ME_DEFAULTS = {
    'n_iterations': 500,
    'batch_size':   10,
    'seed':         42,
    'save_dir':     'results/m4/map_elites/',
    'sim_time':     1.0,
    'n_workers':    4,
}


# ── Top-level worker (picklable for multiprocessing) ──────────────────────────

def _me_eval_worker(args: tuple) -> tuple:
    """Evaluate one genome for MAP-Elites; returns (genome, distance)."""
    genome_arr, sim_time = args
    g = largest_connected_component(np.asarray(genome_arr, dtype=int))
    if int(np.sum(g > 0)) < 8 or int(np.sum(g >= 2)) == 0:
        return (g, 0.0)
    distance = evaluate_genome(g, sim_time=sim_time)
    return (g, float(distance))


def compute_behaviour(genome: np.ndarray, fitness: float) -> tuple:
    """Map a genome to its (distance_bin, symmetry_bin) behaviour descriptor.

    Args:
        genome:  (8,8,8) int array
        fitness: locomotion distance (float in [0, ~1])
    Returns:
        (distance_bin, symmetry_bin) — each an int in [0, 9]
    """
    distance_bin = min(int(fitness * 10), 9)
    symmetry_bin = min(int(bilateral_symmetry_score(genome) * 10), 9)
    return (distance_bin, symmetry_bin)


def run_map_elites(config: dict = None) -> tuple:
    """Run MAP-Elites to fill a 10×10 behaviour grid.

    Algorithm:
        1. Initialise grid with batch_size random genomes (parallel eval).
        2. Each iteration: randomly select batch_size genomes from filled cells,
           mutate each, evaluate the batch in parallel, place winners in grid.
        3. Saves grid every 100 iterations.

    Args:
        config: dict of overrides for ME_DEFAULTS
    Returns:
        (grid, logbook)
        grid:    dict mapping (dist_bin, sym_bin) → (genome_array, fitness)
        logbook: list of dicts with per-iteration stats
    """
    cfg = {**ME_DEFAULTS, **(config or {})}
    n_iterations = cfg['n_iterations']
    batch_size   = cfg['batch_size']
    seed         = cfg['seed']
    save_dir     = cfg['save_dir']
    sim_time     = cfg['sim_time']
    n_workers    = cfg['n_workers']

    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # grid[cell] = (genome_array, fitness)
    grid: dict = {}
    logbook: list = []

    def _place_results(results: list) -> None:
        """Place (genome, distance) pairs into grid (serial — not thread-safe to parallelise)."""
        for g, distance in results:
            cell = compute_behaviour(g, distance)
            current = grid.get(cell)
            if current is None or distance > current[1]:
                grid[cell] = (g.copy(), distance)

    def _eval_batch(genomes: list) -> list:
        """Evaluate a list of genomes in parallel; return (genome, distance) pairs."""
        args = [(np.asarray(g, dtype=int), sim_time) for g in genomes]
        with Pool(processes=n_workers) as pool:
            return pool.map(_me_eval_worker, args)

    # ── Initialisation: evaluate initial random population in parallel ────────
    init_genomes = [random_genome(rng) for _ in range(batch_size)]
    _place_results(_eval_batch(init_genomes))
    _snapshot_log(logbook, iteration=0, grid=grid)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for it in range(1, n_iterations + 1):
        cells = list(grid.keys())
        chosen_cells = [cells[int(rng.integers(0, len(cells)))]
                        for _ in range(batch_size)]

        # Mutate all parents serially (rng state must be deterministic)
        batch = []
        for cell in chosen_cells:
            parent_genome, _ = grid[cell]
            tmp = parent_genome.copy().view(np.ndarray)
            (tmp,) = mutate_random_flip(tmp, rng=rng)
            batch.append(tmp)

        # Evaluate the batch in parallel, then update grid serially
        _place_results(_eval_batch(batch))

        if it % 100 == 0:
            _snapshot_log(logbook, iteration=it, grid=grid)
            _save_grid(grid, save_dir, tag=f'iter_{it:05d}')

    # Final save
    _snapshot_log(logbook, iteration=n_iterations, grid=grid)
    _save_grid(grid, save_dir, tag='final')

    with open(os.path.join(save_dir, 'log.json'), 'w') as f:
        json.dump(logbook, f, indent=2)

    return grid, logbook


def _snapshot_log(logbook: list, iteration: int, grid: dict) -> None:
    fitnesses = [v[1] for v in grid.values()]
    logbook.append({
        'iteration':     iteration,
        'n_filled_cells': len(grid),
        'max_fitness':   float(max(fitnesses)) if fitnesses else 0.0,
        'mean_fitness':  float(np.mean(fitnesses)) if fitnesses else 0.0,
    })


def _save_grid(grid: dict, save_dir: str, tag: str = '') -> None:
    arr = np.empty(1, dtype=object)
    arr[0] = {k: (v[0], v[1]) for k, v in grid.items()}
    path = os.path.join(save_dir, f'grid{"_" + tag if tag else ""}.npy')
    np.save(path, arr, allow_pickle=True)


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_map_elites_grid(
    grid: dict,
    save_path: str = 'results/m4/map_elites/grid_heatmap.png',
) -> None:
    """10×10 heatmap of MAP-Elites fitness values.

    Args:
        grid:      dict mapping (dist_bin, sym_bin) → (genome, fitness)
        save_path: output PNG path
    """
    import matplotlib.pyplot as plt

    heatmap = np.full((10, 10), np.nan)
    for (dist_bin, sym_bin), (_, fitness) in grid.items():
        heatmap[dist_bin, sym_bin] = fitness

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        heatmap, origin='lower', cmap='plasma',
        vmin=0.0, vmax=np.nanmax(heatmap) if not np.all(np.isnan(heatmap)) else 1.0,
    )
    # White for empty cells
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='white')
    ax.images[0].set_cmap(cmap)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Fitness (NormFinalDist)')
    ax.set_xlabel('Symmetry bins')
    ax.set_ylabel('Distance bins')
    ax.set_title('MAP-Elites — Behaviour Space Coverage')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved MAP-Elites grid heatmap → {save_path}')


def compare_map_elites_vs_nsga2(
    me_grid: dict,
    nsga2_population: list,
    save_path: str = 'results/m4/me_vs_nsga2.png',
) -> None:
    """Side-by-side comparison of MAP-Elites and NSGA-II coverage.

    Args:
        me_grid:          dict from run_map_elites()
        nsga2_population: list of IndividualMO from run_nsga2()
        save_path:        output PNG path
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    me_cells_filled = len(me_grid)
    me_max_fitness  = max((v[1] for v in me_grid.values()), default=0.0)

    # Count unique behaviour cells for NSGA-II population
    nsga2_cells = set()
    nsga2_max   = 0.0
    for ind in nsga2_population:
        if not ind.fitness.valid:
            continue
        dist = ind.fitness.values[0]
        cell = compute_behaviour(np.asarray(ind, dtype=int), dist)
        nsga2_cells.add(cell)
        if dist > nsga2_max:
            nsga2_max = dist

    nsga2_cells_filled = len(nsga2_cells)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Subplot 1: Unique behaviours
    ax = axes[0]
    ax.bar(['MAP-Elites', 'NSGA-II'], [me_cells_filled, nsga2_cells_filled],
           color=['purple', 'steelblue'])
    ax.set_ylabel('Number of unique behaviours')
    ax.set_title('Behaviour coverage')

    # Subplot 2: Max fitness
    ax = axes[1]
    ax.bar(['MAP-Elites', 'NSGA-II'], [me_max_fitness, nsga2_max],
           color=['purple', 'steelblue'])
    ax.set_ylabel('Max fitness (NormFinalDist)')
    ax.set_title('Best fitness found')

    # Subplot 3: Coverage heatmaps (cells filled by each method)
    ax = axes[2]
    me_grid_map   = np.zeros((10, 10))
    nsga2_grid_map = np.zeros((10, 10))
    for (d, s) in me_grid.keys():
        me_grid_map[d, s] = 1
    for (d, s) in nsga2_cells:
        nsga2_grid_map[d, s] = 2
    # Overlay: 0=empty, 1=ME only, 2=NSGA2 only, 3=both
    combined = me_grid_map + nsga2_grid_map
    im = ax.imshow(combined, origin='lower', cmap='RdYlGn', vmin=0, vmax=3)
    ax.set_xlabel('Symmetry bins')
    ax.set_ylabel('Distance bins')
    ax.set_title('Cell coverage (green=both, yellow=one)')

    fig.suptitle('MAP-Elites vs NSGA-II Comparison', fontsize=13)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved ME vs NSGA-II comparison → {save_path}')


if __name__ == '__main__':
    grid, log = run_map_elites()
    print(f'MAP-Elites done. Cells filled: {len(grid)} / 100')
    plot_map_elites_grid(grid)
