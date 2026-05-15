"""DEAP evolutionary algorithm for 8×8×8 voxel soft robots.

Public API
----------
run_evolution(...)  →  (population, logbook, hof)
diversity(pop)      →  float
toolbox             →  pre-configured DEAP Toolbox (defaults only)
stats               →  DEAP Statistics object
"""

import copy
import functools

import numpy as np
from deap import algorithms, base, creator, tools
from scipy.ndimage import binary_dilation, binary_erosion

from src.milestone1.fitness import evaluate_genome as _sim
from src.milestone1.genome import largest_connected_component, random_genome

# ── DEAP type creation (guarded against double-registration on re-import) ──
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

N_MATERIALS = 4
GRID = (8, 8, 8)


# ── Fitness wrapper ─────────────────────────────────────────────────────────

def _eval(individual, sim_time: float = 0.5) -> tuple:
    """Evaluate an individual; apply LCC first. Returns DEAP fitness tuple."""
    g = largest_connected_component(np.asarray(individual, dtype=int))
    return (_sim(g, sim_time=sim_time),)


# ── Mutation operators ──────────────────────────────────────────────────────

def mutate_random_flip(individual, flip_prob: float = 0.05, rng=None):
    """Flip each voxel to a random material with probability flip_prob.

    Args:
        individual: DEAP Individual (8×8×8 int array, modified in-place)
        flip_prob:  per-voxel mutation probability
        rng:        optional numpy Generator for reproducibility
    Returns:
        (individual,) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(individual.shape) < flip_prob
    if mask.any():
        individual[mask] = rng.integers(0, N_MATERIALS, size=int(mask.sum()))
    individual[:] = largest_connected_component(individual)
    return (individual,)


def mutate_block(individual, block_size: int = 2, rng=None):
    """Replace a random block_size³ sub-volume with uniformly random materials.

    Args:
        individual: DEAP Individual (modified in-place)
        block_size: side length of the cubic sub-volume
        rng:        optional numpy Generator
    Returns:
        (individual,) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    sx, sy, sz = individual.shape
    x0 = int(rng.integers(0, sx - block_size + 1))
    y0 = int(rng.integers(0, sy - block_size + 1))
    z0 = int(rng.integers(0, sz - block_size + 1))
    individual[x0:x0+block_size, y0:y0+block_size, z0:z0+block_size] = \
        rng.integers(0, N_MATERIALS, size=(block_size, block_size, block_size))
    individual[:] = largest_connected_component(individual)
    return (individual,)


def mutate_grow_shrink(individual, prob_grow: float = 0.3,
                       prob_shrink: float = 0.3, rng=None):
    """Dilate or erode the body surface by one voxel layer.

    With prob_grow: add surface voxels (random non-empty material).
    With prob_shrink: remove all voxels on the current surface.
    Otherwise: do nothing.

    Args:
        individual:  DEAP Individual (modified in-place)
        prob_grow:   probability of grow step
        prob_shrink: probability of shrink step
        rng:         optional numpy Generator
    Returns:
        (individual,) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    r = float(rng.random())
    occupied = individual > 0
    if r < prob_grow:
        new_voxels = binary_dilation(occupied) & ~occupied
        if new_voxels.any():
            n = int(new_voxels.sum())
            individual[new_voxels] = rng.integers(1, N_MATERIALS, size=n)
    elif r < prob_grow + prob_shrink:
        surface = occupied & ~binary_erosion(occupied)
        # Only shrink if at least one active voxel survives — avoids wasting a sim call
        if int(np.sum((individual > 0) & ~surface & (individual >= 2))) > 0:
            individual[surface] = 0
    individual[:] = largest_connected_component(individual)
    return (individual,)


# ── Crossover operators ─────────────────────────────────────────────────────

def cx_uniform_voxel(ind1, ind2, indpb: float = 0.5, rng=None):
    """Swap each voxel between ind1 and ind2 independently with prob indpb.

    Args:
        ind1, ind2: DEAP Individuals (modified in-place)
        indpb:      per-voxel swap probability
        rng:        optional numpy Generator
    Returns:
        (ind1, ind2) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(ind1.shape) < indpb
    tmp = ind1[mask].copy()
    ind1[mask] = ind2[mask]
    ind2[mask] = tmp
    # varOr only keeps ind1 — apply LCC to ind1 only; ind2 is discarded by varOr
    ind1[:] = largest_connected_component(ind1)
    ind2[:] = largest_connected_component(ind2)  # kept for direct use outside varOr
    return ind1, ind2


def cx_one_point_slice(ind1, ind2, rng=None):
    """Swap all voxels at z ≥ z_cut between ind1 and ind2 (Z-axis slab cut).

    Args:
        ind1, ind2: DEAP Individuals (modified in-place)
        rng:        optional numpy Generator
    Returns:
        (ind1, ind2) tuple
    """
    if rng is None:
        rng = np.random.default_rng()
    z_cut = int(rng.integers(1, ind1.shape[2]))   # 1..7 inclusive
    tmp = ind1[:, :, z_cut:].copy()
    ind1[:, :, z_cut:] = ind2[:, :, z_cut:]
    ind2[:, :, z_cut:] = tmp
    ind1[:] = largest_connected_component(ind1)
    ind2[:] = largest_connected_component(ind2)
    return ind1, ind2


# ── Default toolbox (used for quick experiments) ────────────────────────────

toolbox = base.Toolbox()
toolbox.register("individual", lambda: creator.Individual(random_genome()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   functools.partial(_eval, sim_time=0.5))
toolbox.register("mate",       cx_uniform_voxel)
toolbox.register("mutate",     mutate_random_flip)
toolbox.register("select",     tools.selTournament, tournsize=2)

# ── Statistics ───────────────────────────────────────────────────────────────

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("max",  np.max)
stats.register("mean", np.mean)
stats.register("std",  np.std)


def diversity(population) -> float:
    """Mean pairwise Hamming distance across population (0=identical, 1=max diverse).

    Args:
        population: list of DEAP Individuals
    Returns:
        float in [0, 1]
    """
    flat = np.array([np.asarray(ind).flatten() for ind in population])
    n = len(flat)
    if n < 2:
        return 0.0
    # Vectorised: compare all pairs at once via broadcasting — O(n²) but no Python loop
    diffs = (flat[:, None, :] != flat[None, :, :]).mean(axis=2)  # (n, n) hamming matrix
    upper = diffs[np.triu_indices(n, k=1)]                        # upper triangle only
    return float(upper.mean())


# ── Operator registry ────────────────────────────────────────────────────────

MUT_OPS = {
    "random_flip": mutate_random_flip,
    "block":       mutate_block,
    "grow_shrink": mutate_grow_shrink,
}


# ── Main evolution driver ────────────────────────────────────────────────────

def run_evolution(
    pop_size: int = 20,
    n_gen: int = 30,
    lambda_: int = 40,
    cxpb: float = 0.5,
    mutpb: float = 0.3,
    tournament_k: int = 2,
    mut_op: str = "random_flip",
    sim_time: float = 0.5,
    seed: int = 42,
    n_workers: int = 4,
    verbose: bool = True,
) -> tuple:
    """Run (μ+λ) EA; return (population, logbook, hof).

    Uses a thread pool for parallel fitness evaluation (each eval calls
    voxelyze as a subprocess, so threads are not GIL-bound).

    Args:
        pop_size:    μ — survivors kept each generation
        n_gen:       number of generations
        lambda_:     offspring produced per generation
        cxpb:        crossover probability applied in varOr
        mutpb:       mutation probability applied in varOr
        tournament_k: tournament size for selection
        mut_op:      mutation operator: 'random_flip' | 'block' | 'grow_shrink'
        sim_time:    voxelyze simulation duration per call (seconds)
        seed:        random seed
        n_workers:   parallel threads for evaluation
        verbose:     print progress every 10 generations

    Returns:
        (population, logbook, hof)  where logbook includes 'diversity' column
    """
    from multiprocessing.pool import ThreadPool

    rng = np.random.default_rng(seed)

    # Fresh toolbox — avoids mutating module-level state across calls
    tb = base.Toolbox()
    tb.register("evaluate", functools.partial(_eval, sim_time=sim_time))
    tb.register("mate",     cx_uniform_voxel)
    tb.register("mutate",   MUT_OPS[mut_op])
    tb.register("select",   tools.selTournament, tournsize=tournament_k)

    pool = ThreadPool(n_workers)
    tb.register("map", pool.map)

    try:
        # ── Initialise population ──────────────────────────────────────────
        pop = [
            creator.Individual(random_genome(np.random.default_rng(seed * 1000 + i)))
            for i in range(pop_size)
        ]
        hof = tools.HallOfFame(10, similar=lambda a, b: (np.asarray(a) == np.asarray(b)).all())

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "max", "mean", "std", "diversity"]

        # ── Evaluate generation 0 ─────────────────────────────────────────
        fitnesses = list(tb.map(tb.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        rec = stats.compile(pop)
        rec.update(gen=0, nevals=len(pop), diversity=diversity(pop))
        logbook.record(**rec)
        if verbose:
            _log_line(rec)

        # ── Generational loop ─────────────────────────────────────────────
        for gen in range(1, n_gen + 1):
            offspring = algorithms.varOr(pop, tb, lambda_, cxpb, mutpb)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(tb.map(tb.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            hof.update(offspring)
            pop = tb.select(pop + offspring, pop_size)

            rec = stats.compile(pop)
            rec.update(gen=gen, nevals=len(invalid), diversity=diversity(pop))
            logbook.record(**rec)
            if verbose and gen % 10 == 0:
                _log_line(rec)

    finally:
        pool.close()
        pool.join()

    return pop, logbook, hof


def _log_line(rec: dict) -> None:
    print(f"gen {rec['gen']:3d} | nevals={rec['nevals']:3d} | "
          f"max={rec['max']:.4f}  mean={rec['mean']:.4f}  "
          f"div={rec['diversity']:.4f}")
