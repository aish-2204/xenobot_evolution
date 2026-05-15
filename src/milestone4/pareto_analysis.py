"""src/milestone4/pareto_analysis.py — Pareto front plots and comparisons."""

import os

import matplotlib.pyplot as plt
import numpy as np

from src.multiobjective import (
    compute_hypervolume,
    extract_pareto_front,
    find_knee_point,
    random_search_baseline,
)


def plot_pareto_front(
    population: list,
    save_path: str = 'results/m4/pareto_front.png',
) -> None:
    """Scatter plot of the Pareto front with annotated extremes.

    Args:
        population: list of IndividualMO with valid fitness values
        save_path:  output PNG path
    """
    pareto = extract_pareto_front(population)
    pareto_set = set(id(ind) for ind in pareto)
    knee = find_knee_point(pareto)

    all_x, all_y, colors = [], [], []
    for ind in population:
        v, d = ind.fitness.values[1], ind.fitness.values[0]
        all_x.append(v)
        all_y.append(d)
        colors.append('purple' if id(ind) in pareto_set else 'grey')

    fig, ax = plt.subplots(figsize=(8, 6))
    for x, y, c in zip(all_x, all_y, colors):
        ax.scatter(x, y, color=c, alpha=0.6, s=40)

    # Legend proxies
    ax.scatter([], [], color='purple', label='Pareto front', s=40)
    ax.scatter([], [], color='grey',   label='Dominated',    s=40, alpha=0.6)

    if population:
        # Most mobile (highest distance)
        most_mobile = max(population, key=lambda ind: ind.fitness.values[0])
        ax.scatter(
            most_mobile.fitness.values[1], most_mobile.fitness.values[0],
            marker='*', color='blue', s=200, zorder=5,
            label=f'Most mobile (d={most_mobile.fitness.values[0]:.3f})',
        )
        # Most efficient (lowest voxels)
        most_efficient = min(population, key=lambda ind: ind.fitness.values[1])
        ax.scatter(
            most_efficient.fitness.values[1], most_efficient.fitness.values[0],
            marker='D', color='red', s=100, zorder=5,
            label=f'Most efficient (v={most_efficient.fitness.values[1]:.0f})',
        )

    if knee is not None:
        ax.scatter(
            knee.fitness.values[1], knee.fitness.values[0],
            marker='o', color='orange', s=160, zorder=6,
            label=f'Knee point',
        )
        ax.annotate(
            'Knee', (knee.fitness.values[1], knee.fitness.values[0]),
            textcoords='offset points', xytext=(6, 6), fontsize=9,
        )

    ax.set_xlabel('Voxel count (fewer = more efficient)')
    ax.set_ylabel('Locomotion distance')
    ax.set_title('NSGA-II Pareto Front — Xenobot Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved pareto front plot → {save_path}')


def render_pareto_extremes(
    population: list,
    save_dir: str = 'renders/m4/',
) -> None:
    """Render the three extreme robots on the Pareto front.

    Saves: most_mobile.png, most_efficient.png, knee_point.png

    Args:
        population: list of IndividualMO with valid fitness values
        save_dir:   output directory for PNGs
    """
    import pyvista as pv
    from src.Visualize import render_genome

    pv.OFF_SCREEN = True
    os.makedirs(save_dir, exist_ok=True)

    most_mobile   = max(population, key=lambda ind: ind.fitness.values[0])
    most_efficient = min(population, key=lambda ind: ind.fitness.values[1])
    knee          = find_knee_point(extract_pareto_front(population))

    cases = [
        (most_mobile,    'most_mobile',
         f'Most Mobile  d={most_mobile.fitness.values[0]:.4f}'),
        (most_efficient, 'most_efficient',
         f'Most Efficient  v={most_efficient.fitness.values[1]:.0f}'),
    ]
    if knee is not None:
        cases.append((
            knee, 'knee_point',
            f'Knee Point  d={knee.fitness.values[0]:.4f}  v={knee.fitness.values[1]:.0f}',
        ))

    for ind, name, title in cases:
        path = os.path.join(save_dir, f'{name}.png')
        render_genome(np.asarray(ind, dtype=int), title=title, save_path=path)
        print(f'Saved {path}')


def compare_hypervolume(
    nsga2_log: list,
    n_evaluations: int,
    save_path: str = 'results/m4/hypervolume_comparison.png',
) -> None:
    """Plot NSGA-II hypervolume curve vs random search baseline.

    Args:
        nsga2_log:     list of per-generation dicts from run_nsga2()
        n_evaluations: how many random genomes to evaluate for baseline
        save_path:     output PNG path
    """
    # NSGA-II curve — cumulative evaluations on x-axis
    cum_evals, hvs = [], []
    total = 0
    for rec in nsga2_log:
        total += rec.get('nevals', 0)
        cum_evals.append(total)
        hvs.append(rec['hypervolume'])

    # Random search baseline — evaluate once, treat as a flat line
    rs_results = random_search_baseline(n_evaluations)

    import moocore
    ref = [0.0, 513.0]
    if rs_results:
        rs_data = np.array(rs_results)
        try:
            rs_hv = float(moocore.hypervolume(rs_data, ref=ref, maximise=[True, False]))
        except Exception:
            rs_hv = 0.0
    else:
        rs_hv = 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cum_evals, hvs, color='purple', linewidth=2, label='NSGA-II')
    ax.axhline(rs_hv, color='grey', linestyle='--', linewidth=1.5,
               label=f'Random search ({n_evaluations} evals, HV={rs_hv:.4f})')

    ax.set_xlabel('Number of fitness evaluations')
    ax.set_ylabel('Hypervolume')
    ax.set_title('NSGA-II vs Random Search — Hypervolume over Evaluations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved hypervolume comparison → {save_path}')
