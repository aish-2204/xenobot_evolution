"""src/milestone3/scaled_run.py

Full EA run with Mac CPU performance defaults.

Entry point: run_full_evolution(config=None) → (population, logbook, hof)
Config keys override DEFAULTS. Uses multiprocessing.Pool for parallel eval.
"""

import json
import os
import pickle
from multiprocessing.pool import ThreadPool
from typing import Optional

import numpy as np
from deap import algorithms, base, creator, tools

from src.ea import MUT_OPS, cx_uniform_voxel, diversity, stats as deap_stats
from src.milestone1.genome import random_genome
from src.voxcraft_runner import evaluate_genome_m3

if 'FitnessMax' not in creator.__dict__:
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

DEFAULTS = {
    'pop_size':     30,
    'n_generations': 100,
    'mu':           30,
    'lambda_':      60,
    'cxpb':         0.4,
    'mutpb':        0.5,
    'tournsize':    7,
    'hof_size':     10,
    'seed':         42,
    'n_workers':    os.cpu_count(),
    'mutation_op':  'random_flip',
    'save_dir':     'results/m3/',
    'sim_time':     1.0,
}


# ── Pool worker (must be module-level to be picklable) ────────────────────────

def _pool_eval_worker(args: tuple) -> tuple:
    """Receives (plain_numpy_array, sim_time); returns (fitness,) tuple."""
    genome_array, sim_time = args
    return evaluate_genome_m3(genome_array, sim_time)


def _hof_similar(a, b) -> bool:
    """Module-level similar fn for HallOfFame — must be picklable for checkpointing."""
    return bool((np.asarray(a) == np.asarray(b)).all())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_gen(gen: int, rec: dict) -> None:
    div = rec.get('diversity', 0.0)
    print(f"Gen {gen} | Max: {rec['max']:.4f} | Mean: {rec['mean']:.4f} | Div: {div:.3f}")


def _save_checkpoint(save_dir: str, gen: int, pop, logbook, hof) -> None:
    path = os.path.join(save_dir, f'checkpoint_gen_{gen}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'gen': gen, 'pop': pop, 'logbook': logbook, 'hof': hof}, f)


def _save_final(save_dir: str, logbook, hof) -> None:
    rows = []
    for rec in logbook:
        rows.append({k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                     for k, v in rec.items()})
    with open(os.path.join(save_dir, 'full_run_log.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    if hof:
        np.save(
            os.path.join(save_dir, 'hof_genomes.npy'),
            np.array([np.asarray(ind, dtype=int) for ind in hof]),
        )
        best = max(hof, key=lambda ind: ind.fitness.values[0])
        np.save(os.path.join(save_dir, 'best_genome.npy'), np.asarray(best, dtype=int))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_full_evolution(config: Optional[dict] = None) -> tuple:
    """Run (mu+lambda) EA with Mac-appropriate defaults.

    Args:
        config: dict of overrides for DEFAULTS (any subset of keys)
    Returns:
        (population, logbook, hof)
    """
    cfg = {**DEFAULTS, **(config or {})}

    save_dir     = cfg['save_dir']
    seed         = cfg['seed']
    n_workers    = cfg['n_workers']
    pop_size     = cfg['pop_size']
    n_generations = cfg['n_generations']
    mu           = cfg['mu']
    lambda_      = cfg['lambda_']
    cxpb         = cfg['cxpb']
    mutpb        = cfg['mutpb']
    tournsize    = cfg['tournsize']
    hof_size     = cfg['hof_size']
    mutation_op  = cfg['mutation_op']
    sim_time     = cfg.get('sim_time', 1.0)

    os.makedirs(save_dir, exist_ok=True)

    tb = base.Toolbox()
    tb.register('evaluate', evaluate_genome_m3)
    tb.register('mate',     cx_uniform_voxel)
    tb.register('mutate',   MUT_OPS[mutation_op])
    tb.register('select',   tools.selTournament, tournsize=tournsize)

    pool = ThreadPool(n_workers)

    # Wrap pool.map to convert Individuals → plain numpy before serialising
    def _pool_map(fn, population):
        tasks = [(np.asarray(ind, dtype=int), sim_time) for ind in population]
        return pool.map(_pool_eval_worker, tasks)

    tb.register('map', _pool_map)

    hof = tools.HallOfFame(hof_size, similar=_hof_similar)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'max', 'mean', 'std', 'diversity']

    pop = [
        creator.Individual(random_genome(np.random.default_rng(seed * 1000 + i)))
        for i in range(pop_size)
    ]

    try:
        # Generation 0
        fitnesses = list(tb.map(tb.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        rec = deap_stats.compile(pop)
        rec.update(gen=0, nevals=len(pop), diversity=diversity(pop))
        logbook.record(**rec)
        _print_gen(0, rec)

        for gen in range(1, n_generations + 1):
            offspring = algorithms.varOr(pop, tb, lambda_, cxpb, mutpb)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(tb.map(tb.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)
            pop = tb.select(pop + offspring, mu)

            rec = deap_stats.compile(pop)
            rec.update(gen=gen, nevals=len(invalid), diversity=diversity(pop))
            logbook.record(**rec)

            if gen % 5 == 0:
                _print_gen(gen, rec)

            if gen % 10 == 0 or gen == n_generations:
                _save_checkpoint(save_dir, gen, pop, logbook, hof)

    finally:
        pool.close()
        pool.join()

    _save_final(save_dir, logbook, hof)
    return pop, logbook, hof


if __name__ == '__main__':
    pop, log, hof = run_full_evolution(config={
        'pop_size': 20, 'n_generations': 10,
        'n_workers': 4, 'save_dir': 'results/m3/',
    })
    print(f'Best: {hof[0].fitness.values[0]:.4f}')
