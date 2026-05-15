"""src/multiobjective.py — NSGA-II multi-objective evolution for xenobots.

Objectives: maximise locomotion distance (weight=+1), minimise voxel count (weight=-1).
"""

import json
import os
from multiprocessing.pool import ThreadPool

import numpy as np
from deap import algorithms, base, creator, tools

from src.ea import MUT_OPS, cx_uniform_voxel
from src.milestone1.fitness import evaluate_genome
from src.milestone1.genome import EMPTY, largest_connected_component, random_genome

# ── DEAP type creation (guarded against double-registration on re-import) ─────
if 'FitnessMO' not in creator.__dict__:
    creator.create('FitnessMO', base.Fitness, weights=(1.0, -1.0))
if 'IndividualMO' not in creator.__dict__:
    creator.create('IndividualMO', np.ndarray, fitness=creator.FitnessMO)

NSGA2_DEFAULTS = {
    'pop_size':      40,
    'n_generations': 80,
    'mu':            40,
    'lambda_':       80,
    'cxpb':          0.5,
    'mutpb':         0.3,
    'seed':          42,
    'n_workers':     4,
    'save_dir':      'results/m4/',
    'sim_time':      1.0,
    'hof_size':      20,
}

REF_POINT = [0.0, 513.0]


# ── Module-level worker (top-level for picklability) ──────────────────────────

def _mo_eval_worker(args: tuple) -> tuple:
    """Evaluate a plain numpy genome array; return (distance, n_voxels)."""
    genome_array, sim_time = args
    genome = largest_connected_component(np.asarray(genome_array, dtype=int))
    if int(np.sum(genome > 0)) < 8 or int(np.sum(genome >= 2)) == 0:
        return (0.0, 0.0)
    distance = evaluate_genome(genome, sim_time=sim_time)
    n_voxels = float((genome != EMPTY).sum())
    return (distance, n_voxels)


# ── Fitness function ──────────────────────────────────────────────────────────

def evaluate_mo(genome: np.ndarray, sim_time: float = 1.0) -> tuple:
    """Evaluate genome on two objectives.

    Args:
        genome:   (8,8,8) int numpy array
        sim_time: simulation duration in seconds (passed to voxelyze)
    Returns:
        (distance, n_voxels) — distance maximised, n_voxels minimised.
        Returns (0.0, 0.0) for invalid or empty genomes.
    """
    genome = largest_connected_component(np.asarray(genome, dtype=int))
    if int(np.sum(genome > 0)) < 8 or int(np.sum(genome >= 2)) == 0:
        return (0.0, 0.0)
    distance = evaluate_genome(genome, sim_time=sim_time)
    n_voxels = float((genome != EMPTY).sum())
    return (distance, n_voxels)


# ── Toolbox builder ───────────────────────────────────────────────────────────

def _make_individual_mo() -> 'creator.IndividualMO':
    return creator.IndividualMO(random_genome())


def build_toolbox_mo(mutation_op: str = 'random_flip', seed: int = 42) -> base.Toolbox:
    """Build and return a configured DEAP Toolbox for NSGA-II.

    Args:
        mutation_op: 'random_flip' | 'block' | 'grow_shrink'
        seed:        numpy random seed
    Returns:
        Configured base.Toolbox with MO operators
    """
    np.random.seed(seed)
    tb = base.Toolbox()
    tb.register('individual', _make_individual_mo)
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('evaluate',   evaluate_mo)
    tb.register('mate',       cx_uniform_voxel)
    tb.register('mutate',     MUT_OPS[mutation_op])
    tb.register('select',     tools.selNSGA2)
    return tb


# ── Pareto-front helpers ──────────────────────────────────────────────────────

def extract_pareto_front(population: list) -> list:
    """Return the non-dominated (first) front from population.

    Args:
        population: list of IndividualMO with valid fitness values
    Returns:
        list of non-dominated individuals
    """
    if not population:
        return []
    return tools.sortNondominated(population, len(population), first_front_only=True)[0]


def find_knee_point(pareto_front: list):
    """Find individual closest to the ideal point in normalised objective space.

    Normalises distance to [0,1] and n_voxels to [0,1], then finds the
    individual closest (Euclidean) to the ideal corner (1.0 distance, 0.0 voxels).

    Args:
        pareto_front: list of non-dominated IndividualMO
    Returns:
        Single IndividualMO knee-point individual, or None if front is empty
    """
    if not pareto_front:
        return None
    fitnesses = np.array([ind.fitness.values for ind in pareto_front])
    d_vals = fitnesses[:, 0]
    v_vals = fitnesses[:, 1]

    d_range = d_vals.max() - d_vals.min()
    v_range = v_vals.max() - v_vals.min()
    d_norm = (d_vals - d_vals.min()) / (d_range if d_range > 0 else 1.0)
    v_norm = (v_vals - v_vals.min()) / (v_range if v_range > 0 else 1.0)

    # Ideal point: maximum distance (1.0), minimum voxels (0.0) after normalisation
    dists_to_ideal = np.sqrt((d_norm - 1.0) ** 2 + v_norm ** 2)
    return pareto_front[int(np.argmin(dists_to_ideal))]


def compute_hypervolume(population: list, ref_point: list = None) -> float:
    """Compute hypervolume indicator for a population.

    Uses moocore.hypervolume (the backend used by DEAP) with maximise=[True, False]
    for (distance, n_voxels) objectives.

    Args:
        population: list of IndividualMO with valid fitness values
        ref_point:  [worst_distance, worst_n_voxels], default [0.0, 513.0]
    Returns:
        float hypervolume dominated by the population
    """
    import moocore
    if ref_point is None:
        ref_point = REF_POINT
    valid = [ind for ind in population if ind.fitness.valid and ind.fitness.values != (0.0, 0.0)]
    if not valid:
        return 0.0
    data = np.array([ind.fitness.values for ind in valid])
    try:
        return float(moocore.hypervolume(data, ref=ref_point, maximise=[True, False]))
    except Exception:
        return 0.0


def random_search_baseline(n_evaluations: int, seed: int = 42) -> list:
    """Evaluate n_evaluations random genomes; used for hypervolume comparison.

    Args:
        n_evaluations: number of random genomes to evaluate
        seed:          random seed
    Returns:
        list of (distance, n_voxels) tuples
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_evaluations):
        genome = random_genome(rng)
        genome = largest_connected_component(genome)
        results.append(evaluate_mo(genome))
    return results


# ── Main NSGA-II driver ───────────────────────────────────────────────────────

def run_nsga2(config: dict = None) -> tuple:
    """Run NSGA-II multi-objective evolution.

    Implements (mu+lambda) selection with NSGA-II crowded-comparison ordering.
    Tracks hypervolume per generation. Uses ThreadPool for parallel evaluation
    (voxelyze subprocess calls are not GIL-bound).

    Args:
        config: dict of overrides for NSGA2_DEFAULTS
    Returns:
        (population, logbook, hof) where logbook is a list of dicts
    """
    cfg = {**NSGA2_DEFAULTS, **(config or {})}
    save_dir      = cfg['save_dir']
    seed          = cfg['seed']
    n_workers     = cfg['n_workers']
    pop_size      = cfg['pop_size']
    n_generations = cfg['n_generations']
    mu            = cfg['mu']
    lambda_       = cfg['lambda_']
    cxpb          = cfg['cxpb']
    mutpb         = cfg['mutpb']
    hof_size      = cfg.get('hof_size', 20)
    sim_time      = cfg.get('sim_time', 1.0)

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)

    tb = build_toolbox_mo(seed=seed)

    pool = ThreadPool(n_workers)

    def _pool_map(fn, population):
        tasks = [(np.asarray(ind, dtype=int), sim_time) for ind in population]
        return pool.map(_mo_eval_worker, tasks)

    tb.register('map', _pool_map)

    hof = tools.HallOfFame(
        hof_size,
        similar=lambda a, b: bool((np.asarray(a) == np.asarray(b)).all()),
    )
    logbook = []

    pop = [
        creator.IndividualMO(random_genome(np.random.default_rng(seed * 1000 + i)))
        for i in range(pop_size)
    ]

    try:
        # ── Generation 0 ──────────────────────────────────────────────────────
        fitnesses = list(tb.map(tb.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        pop = tb.select(pop, mu)

        pf = extract_pareto_front(pop)
        hv = compute_hypervolume(pop)
        fitvals = [ind.fitness.values for ind in pop]
        logbook.append({
            'gen': 0,
            'nevals': len(pop),
            'hypervolume': hv,
            'pareto_front_size': len(pf),
            'max_distance': float(max(v[0] for v in fitvals)),
            'min_voxels': float(min(v[1] for v in fitvals)),
        })
        print(f"Gen   0 | HV: {hv:.4f} | Pareto size: {len(pf):3d} | "
              f"Max dist: {logbook[-1]['max_distance']:.4f}")

        # ── Generational loop ──────────────────────────────────────────────────
        for gen in range(1, n_generations + 1):
            offspring = algorithms.varOr(pop, tb, lambda_, cxpb, mutpb)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(tb.map(tb.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)
            pop = tb.select(pop + offspring, mu)

            pf = extract_pareto_front(pop)
            hv = compute_hypervolume(pop)
            fitvals = [ind.fitness.values for ind in pop]
            rec = {
                'gen': gen,
                'nevals': len(invalid),
                'hypervolume': hv,
                'pareto_front_size': len(pf),
                'max_distance': float(max(v[0] for v in fitvals)),
                'min_voxels': float(min(v[1] for v in fitvals)),
            }
            logbook.append(rec)
            if gen % 10 == 0:
                print(f"Gen {gen:3d} | HV: {hv:.4f} | Pareto size: {len(pf):3d} | "
                      f"Max dist: {rec['max_distance']:.4f}")

    finally:
        pool.close()
        pool.join()

    # ── Save outputs ───────────────────────────────────────────────────────────
    with open(os.path.join(save_dir, 'nsga2_log.json'), 'w') as f:
        json.dump(logbook, f, indent=2)

    pop_arr = np.empty(len(pop), dtype=object)
    for i, ind in enumerate(pop):
        pop_arr[i] = np.asarray(ind, dtype=int)
    np.save(os.path.join(save_dir, 'nsga2_final_pop.npy'), pop_arr, allow_pickle=True)

    hof_arr = np.empty(len(hof), dtype=object)
    for i, ind in enumerate(hof):
        hof_arr[i] = np.asarray(ind, dtype=int)
    np.save(os.path.join(save_dir, 'nsga2_hof.npy'), hof_arr, allow_pickle=True)

    return pop, logbook, hof


if __name__ == '__main__':
    pop, log, hof = run_nsga2()
    print(f'Done. Best distance: {max(ind.fitness.values[0] for ind in pop):.4f}')
