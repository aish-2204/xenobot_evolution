"""src/milestone3/ablation.py

Ablation study: 6 conditions × N seeds.

Conditions
----------
baseline          Full EA
no_crossover      cxpb=0
no_connectivity   Skip LCC in mutation and eval
random_selection  selRandom instead of selTournament
single_material   Genomes restricted to materials {0, 1}
no_mutation       mutpb=0

Public API
----------
run_ablation_condition(condition_name, seed, save_dir)  → (logbook, best_fitness)
run_all_ablations(seeds, save_dir)                      → results dict
plot_ablation_results(results_dir, save_path)           → None
"""

import csv
import json
import os
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Tuple

import numpy as np
from deap import algorithms, base, creator, tools
from scipy.stats import mannwhitneyu

from src.ea import MUT_OPS, cx_uniform_voxel, diversity, stats as deap_stats
from src.milestone1.fitness import evaluate_genome
from src.milestone1.genome import largest_connected_component, random_genome

if 'FitnessMax' not in creator.__dict__:
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

ABLATION_CONDITIONS: List[str] = [
    'baseline',
    'no_crossover',
    'no_connectivity',
    'random_selection',
    'single_material',
    'no_mutation',
]

ABLATION_DEFAULTS: dict = {
    'pop_size':     20,
    'n_generations': 50,
    'mu':           20,
    'lambda_':      40,
    'cxpb':         0.4,
    'mutpb':        0.3,
    'tournsize':    7,
    'n_workers':    4,
    'sim_time':     0.5,
}


# ── Condition-specific operator variants ──────────────────────────────────────

def _mutate_no_lcc(individual, flip_prob: float = 0.05, rng=None):
    """random_flip without LCC repair — for no_connectivity condition."""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(individual.shape) < flip_prob
    if mask.any():
        individual[mask] = rng.integers(0, 4, size=int(mask.sum()))
    return (individual,)


def _mutate_single_mat(individual, flip_prob: float = 0.05, rng=None):
    """random_flip capped to materials {0, 1} — for single_material condition."""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(individual.shape) < flip_prob
    if mask.any():
        individual[mask] = rng.integers(0, 2, size=int(mask.sum()))
    individual[individual >= 2] = 1
    individual[:] = largest_connected_component(individual)
    return (individual,)


def _eval_no_lcc(individual, sim_time: float = 0.5) -> Tuple[float]:
    """Evaluate without applying LCC — for no_connectivity condition."""
    return (evaluate_genome(np.asarray(individual, dtype=int), sim_time=sim_time),)


def _eval_standard(individual, sim_time: float = 0.5) -> Tuple[float]:
    """Standard eval: apply LCC then simulate."""
    g = largest_connected_component(np.asarray(individual, dtype=int))
    return (evaluate_genome(g, sim_time=sim_time),)


def _random_genome_single_mat(rng=None) -> np.ndarray:
    """Random genome restricted to materials {0, 1}."""
    if rng is None:
        rng = np.random.default_rng()
    arr = rng.integers(0, 2, size=(8, 8, 8))
    return largest_connected_component(arr)


# ── Per-condition toolbox factory ─────────────────────────────────────────────

def _build_toolbox(condition: str, sim_time: float, tournsize: int) -> base.Toolbox:
    import functools
    tb = base.Toolbox()

    if condition == 'no_connectivity':
        tb.register('evaluate', functools.partial(_eval_no_lcc, sim_time=sim_time))
        tb.register('mutate', _mutate_no_lcc)
    elif condition == 'single_material':
        tb.register('evaluate', functools.partial(_eval_standard, sim_time=sim_time))
        tb.register('mutate', _mutate_single_mat)
    else:
        tb.register('evaluate', functools.partial(_eval_standard, sim_time=sim_time))
        tb.register('mutate', MUT_OPS['random_flip'])

    if condition == 'random_selection':
        tb.register('select', tools.selRandom)
    else:
        tb.register('select', tools.selTournament, tournsize=tournsize)

    tb.register('mate', cx_uniform_voxel)
    return tb


def _init_population(condition: str, pop_size: int, seed: int) -> list:
    inds = []
    for i in range(pop_size):
        rng_i = np.random.default_rng(seed * 1000 + i)
        if condition == 'single_material':
            arr = _random_genome_single_mat(rng_i)
        else:
            arr = random_genome(rng_i)
        inds.append(creator.Individual(arr))
    return inds


# ── Single condition-seed run (internal) ──────────────────────────────────────

def _run_condition(
    condition: str,
    seed: int,
    pop_size: int,
    n_gen: int,
    lambda_: int,
    cxpb: float,
    mutpb: float,
    tournsize: int,
    sim_time: float,
    n_workers: int,
    verbose: bool = True,
) -> Tuple[tools.Logbook, tools.HallOfFame]:
    _cxpb  = 0.0 if condition == 'no_crossover' else cxpb
    _mutpb = 0.0 if condition == 'no_mutation'  else mutpb

    tb   = _build_toolbox(condition, sim_time, tournsize)
    pool = ThreadPool(n_workers)
    tb.register('map', pool.map)

    hof     = tools.HallOfFame(5, similar=lambda a, b: (np.asarray(a) == np.asarray(b)).all())
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'max', 'mean', 'std', 'diversity']

    try:
        pop = _init_population(condition, pop_size, seed)

        fitnesses = list(tb.map(tb.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        rec = deap_stats.compile(pop)
        rec.update(gen=0, nevals=len(pop), diversity=diversity(pop))
        logbook.record(**rec)

        for gen in range(1, n_gen + 1):
            offspring = algorithms.varOr(pop, tb, lambda_, _cxpb, _mutpb)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(tb.map(tb.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)
            pop = tb.select(pop + offspring, pop_size)

            rec = deap_stats.compile(pop)
            rec.update(gen=gen, nevals=len(invalid), diversity=diversity(pop))
            logbook.record(**rec)

            if verbose and gen % 10 == 0:
                print(f"  [{condition:20s}] seed={seed} gen={gen:4d} "
                      f"max={rec['max']:.4f} div={rec['diversity']:.4f}")

    finally:
        pool.close()
        pool.join()

    return logbook, hof


# ── Public API ────────────────────────────────────────────────────────────────

def run_ablation_condition(
    condition_name: str,
    seed: int,
    save_dir: str = 'results/m3/ablation/',
) -> Tuple[tools.Logbook, float]:
    """Run one ablation condition for one seed.

    Args:
        condition_name: one of ABLATION_CONDITIONS
        seed:           random seed
        save_dir:       directory to save logbook JSON
    Returns:
        (logbook, best_fitness_at_final_gen)
    """
    os.makedirs(save_dir, exist_ok=True)
    d = ABLATION_DEFAULTS

    logbook, hof = _run_condition(
        condition=condition_name,
        seed=seed,
        pop_size=d['pop_size'],
        n_gen=d['n_generations'],
        lambda_=d['lambda_'],
        cxpb=d['cxpb'],
        mutpb=d['mutpb'],
        tournsize=d['tournsize'],
        sim_time=d['sim_time'],
        n_workers=d['n_workers'],
        verbose=True,
    )

    out_path = os.path.join(save_dir, f'{condition_name}_seed{seed}.json')
    _save_logbook(logbook, out_path)

    max_vals = logbook.select('max')
    best_fitness = float(max_vals[-1]) if max_vals else 0.0
    return logbook, best_fitness


def run_all_ablations(
    seeds: List[int] = None,
    save_dir: str = 'results/m3/ablation/',
) -> Dict[str, list]:
    """Run all 6 conditions × len(seeds) seeds sequentially.

    Args:
        seeds:    list of random seeds (default [42, 123])
        save_dir: directory for per-run JSON logbooks and summary CSVs
    Returns:
        dict: {condition: [{'seed': int, 'logbook': Logbook, 'best_fitness': float}, ...]}
    """
    if seeds is None:
        seeds = [42, 123]

    os.makedirs(save_dir, exist_ok=True)
    total = len(ABLATION_CONDITIONS) * len(seeds)
    n = 0
    results: Dict[str, list] = {c: [] for c in ABLATION_CONDITIONS}

    for condition in ABLATION_CONDITIONS:
        for seed in seeds:
            n += 1
            print(f'Running {condition} seed {seed} ({n}/{total})...')
            logbook, best = run_ablation_condition(condition, seed, save_dir)
            results[condition].append({
                'seed': seed,
                'logbook': logbook,
                'best_fitness': best,
            })

    # ablation_stats.csv
    with open(os.path.join(save_dir, 'ablation_stats.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'seed', 'best_fitness', 'auc_fitness'])
        for condition, runs in results.items():
            for run in runs:
                max_vals = run['logbook'].select('max')
                auc = float(np.trapz(max_vals)) if max_vals else 0.0
                writer.writerow([
                    condition, run['seed'],
                    f"{run['best_fitness']:.6f}", f"{auc:.6f}",
                ])

    # ablation_mannwhitney.csv
    baseline_bests = [r['best_fitness'] for r in results.get('baseline', [])]
    with open(os.path.join(save_dir, 'ablation_mannwhitney.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['condition', 'p_value'])
        for condition, runs in results.items():
            bests = [r['best_fitness'] for r in runs]
            if condition == 'baseline' or len(baseline_bests) < 2 or len(bests) < 2:
                p = float('nan')
            else:
                _, p = mannwhitneyu(bests, baseline_bests, alternative='two-sided')
            writer.writerow([condition, f'{p:.4f}'])

    print(f'Saved stats → {save_dir}ablation_stats.csv / ablation_mannwhitney.csv')
    return results


def plot_ablation_results(
    results_dir: str = 'results/m3/ablation/',
    save_path: str = 'results/m3/ablation_comparison.png',
) -> None:
    """Plot fitness curves and bar chart of ablation conditions.

    Args:
        results_dir: directory containing per-run JSON logbooks and CSV stats
        save_path:   output PNG path
    """
    import matplotlib.pyplot as plt

    COLORS = {
        'baseline': 'black',
        'no_crossover': 'steelblue',
        'no_connectivity': 'seagreen',
        'random_selection': 'darkorange',
        'single_material': 'mediumpurple',
        'no_mutation': 'crimson',
    }

    # Load logbooks
    cond_data: Dict[str, list] = {c: [] for c in ABLATION_CONDITIONS}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        for cond in ABLATION_CONDITIONS:
            if fname.startswith(cond + '_seed'):
                with open(os.path.join(results_dir, fname)) as f:
                    rows = json.load(f)
                cond_data[cond].append([r['max'] for r in rows])
                break

    # Load p-values
    p_values: Dict[str, float] = {}
    pval_path = os.path.join(results_dir, 'ablation_mannwhitney.csv')
    if os.path.exists(pval_path):
        with open(pval_path) as f:
            for row in csv.DictReader(f):
                try:
                    p_values[row['condition']] = float(row['p_value'])
                except ValueError:
                    p_values[row['condition']] = float('nan')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    final_means: Dict[str, float] = {}
    final_stds: Dict[str, float] = {}

    for cond in ABLATION_CONDITIONS:
        logs = cond_data[cond]
        color = COLORS.get(cond, 'gray')
        if not logs:
            final_means[cond] = 0.0
            final_stds[cond] = 0.0
            continue
        max_len = max(len(l) for l in logs)
        padded = [l + [l[-1]] * (max_len - len(l)) for l in logs]
        arr = np.array(padded)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        gens = np.arange(max_len)
        lw = 2.0 if cond == 'baseline' else 1.2
        ax1.plot(gens, mean, label=cond, color=color, linewidth=lw)
        ax1.fill_between(gens, mean - std, mean + std, alpha=0.12, color=color)
        final_means[cond] = float(mean[-1])
        final_stds[cond]  = float(std[-1])

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best fitness (mean ± std across seeds)')
    ax1.set_title('Fitness curves by ablation condition')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    x      = np.arange(len(ABLATION_CONDITIONS))
    means  = [final_means[c] for c in ABLATION_CONDITIONS]
    stds   = [final_stds[c]  for c in ABLATION_CONDITIONS]
    bcolors = [
        'red' if (not np.isnan(p_values.get(c, float('nan'))) and p_values.get(c, 1.0) < 0.05)
        else 'gray'
        for c in ABLATION_CONDITIONS
    ]
    ax2.bar(x, means, yerr=stds, color=bcolors, capsize=4, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ABLATION_CONDITIONS, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Best fitness at final generation')
    ax2.set_title('Final performance  (red = p<0.05 vs baseline)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved ablation comparison → {save_path}')


# ── Serialisation helper ──────────────────────────────────────────────────────

def _save_logbook(logbook: tools.Logbook, path: str) -> None:
    rows = []
    for rec in logbook:
        rows.append({k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                     for k, v in rec.items()})
    with open(path, 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    run_all_ablations(seeds=[42, 123], save_dir='results/m3/ablation/')
    plot_ablation_results()
