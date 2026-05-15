"""src/milestone4/morphological_analysis.py — Morphological analysis of evolved robots."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.milestone1.genome import ACTIVE_N, ACTIVE_P, EMPTY


# ── Per-genome metrics ────────────────────────────────────────────────────────

def active_passive_ratio(genome: np.ndarray) -> float:
    """Ratio of active voxels to all filled voxels.

    Args:
        genome: (8,8,8) int array
    Returns:
        float in [0, 1]; 0.0 if genome is empty
    """
    genome = np.asarray(genome, dtype=int)
    n_filled = int(np.sum(genome != EMPTY))
    if n_filled == 0:
        return 0.0
    n_active = int(np.sum(genome == ACTIVE_P)) + int(np.sum(genome == ACTIVE_N))
    return n_active / n_filled


def centre_of_mass_height(genome: np.ndarray) -> float:
    """Mean Z-coordinate of filled voxels, normalised to [0, 1] by dividing by 7.0.

    Args:
        genome: (8,8,8) int array
    Returns:
        float in [0, 1]; 0.0 if genome is empty
    """
    genome = np.asarray(genome, dtype=int)
    z_coords = np.argwhere(genome != EMPTY)[:, 2]
    if len(z_coords) == 0:
        return 0.0
    return float(z_coords.mean()) / 7.0


def bilateral_symmetry_score(genome: np.ndarray) -> float:
    """Symmetry score about the X midplane (x=4 boundary).

    Compares genome[:4,:,:] with the mirror of genome[4:,:,:] along X.
    Score = 1 - (n_differing / n_total), range [0, 1].

    Args:
        genome: (8,8,8) int array
    Returns:
        float in [0, 1]; 1.0 = perfectly symmetric
    """
    genome = np.asarray(genome, dtype=int)
    left  = genome[:4, :, :]
    right = genome[4:, :, :]
    right_mirrored = right[::-1, :, :]
    n_total = left.size
    n_differing = int(np.sum(left != right_mirrored))
    return 1.0 - (n_differing / n_total)


# ── Aggregate analysis ────────────────────────────────────────────────────────

def analyse_top_robots(
    hof_genomes: list,
    fitnesses: list,
    save_dir: str = 'results/m4/',
) -> None:
    """Compute morphological metrics for top robots and produce scatter plots.

    Args:
        hof_genomes: list of (8,8,8) numpy arrays
        fitnesses:   list of float fitness scores, same length as hof_genomes
        save_dir:    directory for output PNGs and CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    stats = []
    for i, (genome, fitness) in enumerate(zip(hof_genomes, fitnesses)):
        g = np.asarray(genome, dtype=int)
        stats.append({
            'genome_idx':  i,
            'fitness':     float(fitness),
            'active_ratio': active_passive_ratio(g),
            'com_height':   centre_of_mass_height(g),
            'symmetry':     bilateral_symmetry_score(g),
        })

    idxs      = [s['genome_idx']  for s in stats]
    fits      = [s['fitness']     for s in stats]
    ratios    = [s['active_ratio'] for s in stats]
    heights   = [s['com_height']  for s in stats]
    syms      = [s['symmetry']    for s in stats]

    # ── active_ratio vs fitness ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(ratios, fits, c=syms, cmap='viridis', s=60, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='Symmetry score')
    ax.set_xlabel('Active-to-passive ratio')
    ax.set_ylabel('Fitness (NormFinalDist)')
    ax.set_title('Active Ratio vs Fitness')
    fig.savefig(os.path.join(save_dir, 'active_ratio_vs_fitness.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── CoM height vs fitness ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(heights, fits, c=syms, cmap='viridis', s=60, alpha=0.8)
    fig.colorbar(sc, ax=ax, label='Symmetry score')
    ax.set_xlabel('Centre of mass height (normalised)')
    ax.set_ylabel('Fitness (NormFinalDist)')
    ax.set_title('CoM Height vs Fitness')
    fig.savefig(os.path.join(save_dir, 'com_height_vs_fitness.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Symmetry vs fitness ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(syms, fits, color='steelblue', s=60, alpha=0.8)
    ax.set_xlabel('Bilateral symmetry score')
    ax.set_ylabel('Fitness (NormFinalDist)')
    ax.set_title('Symmetry vs Fitness')
    fig.savefig(os.path.join(save_dir, 'symmetry_vs_fitness.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(save_dir, 'morphology_stats.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['genome_idx', 'fitness', 'active_ratio',
                                                'com_height', 'symmetry'])
        writer.writeheader()
        writer.writerows(stats)

    print(f'Saved morphological analysis to {save_dir}')


def find_structural_motifs(
    genomes: list,
    top_n: int = 20,
    window: int = 2,
) -> list:
    """Find the 5 most frequent window×window×window voxel patterns across top genomes.

    Args:
        genomes: list of (8,8,8) numpy arrays
        top_n:   use only the first top_n genomes
        window:  side length of the sliding cube window
    Returns:
        list of (pattern_array, frequency) tuples, length ≤ 5
    """
    pattern_counts: dict = {}
    for genome in genomes[:top_n]:
        g = np.asarray(genome, dtype=int)
        sx, sy, sz = g.shape
        for x in range(sx - window + 1):
            for y in range(sy - window + 1):
                for z in range(sz - window + 1):
                    key = tuple(g[x:x+window, y:y+window, z:z+window].flatten())
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1

    sorted_patterns = sorted(pattern_counts.items(), key=lambda kv: kv[1], reverse=True)

    os.makedirs('results/m4', exist_ok=True)
    if sorted_patterns:
        top5 = sorted_patterns[:5]
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [str(i + 1) for i in range(len(top5))]
        freqs  = [kv[1] for kv in top5]
        ax.bar(labels, freqs, color='steelblue')
        ax.set_xlabel('Motif rank')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top structural motifs (window={window}×{window}×{window})')
        fig.savefig('results/m4/motif_frequency.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('Saved results/m4/motif_frequency.png')

    return [
        (np.array(k, dtype=int).reshape(window, window, window), freq)
        for k, freq in sorted_patterns[:5]
    ]


def render_morphology_comparison(
    hof_genomes: list,
    fitnesses: list,
    n: int = 5,
    save_dir: str = 'renders/m4/morphology/',
) -> None:
    """Render top-n robots individually and save to save_dir.

    Args:
        hof_genomes: list of (8,8,8) numpy arrays (sorted best-first)
        fitnesses:   list of float fitness scores
        n:           number of robots to render
        save_dir:    output directory
    """
    import pyvista as pv
    from src.Visualize import render_genome

    pv.OFF_SCREEN = True
    os.makedirs(save_dir, exist_ok=True)

    for i, (genome, fitness) in enumerate(zip(hof_genomes[:n], fitnesses[:n])):
        path = os.path.join(save_dir, f'rank_{i}_fitness_{fitness:.4f}.png')
        render_genome(
            np.asarray(genome, dtype=int),
            title=f'Rank {i} | fit={fitness:.4f}',
            save_path=path,
        )
        print(f'Saved {path}')
