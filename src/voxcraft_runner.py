"""src/voxcraft_runner.py

Milestone 3 voxelyze subprocess runner — canonical interface.

Functions
---------
genome_to_vxa         write VXA file to disk
run_voxcraft          run one genome, return (dx, dy, dz) float tuple
evaluate_genome_m3    DEAP-compatible fitness: returns (fitness,) tuple
batch_evaluate_parallel  parallel eval with multiprocessing.Pool
"""

import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np

from src.milestone1.fitness import _parse_fitness
from src.milestone1.genome import largest_connected_component
from src.milestone1.serializer import genome_to_vxa as _serializer_genome_to_vxa


# ── File writer ───────────────────────────────────────────────────────────────

def genome_to_vxa(genome: np.ndarray, output_path: str, sim_time: float = 1.0) -> None:
    """Write genome as VXA XML file to output_path.

    Args:
        genome:      (8,8,8) int array — 0=empty, 1=passive, 2=active+, 3=active-
        output_path: destination file path for the VXA XML
        sim_time:    simulation duration in seconds
    Returns:
        None
    """
    fitness_file = os.path.splitext(os.path.abspath(output_path))[0] + '_output.xml'
    xml = _serializer_genome_to_vxa(genome, sim_time=sim_time, fitness_file=fitness_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(xml)


# ── Binary locator ────────────────────────────────────────────────────────────

def _find_binary() -> str:
    """Locate voxelyze binary: VOXCRAFT_BIN env var → ./voxelyze → PATH."""
    env = os.environ.get('VOXCRAFT_BIN', '')
    if env and os.path.isfile(env):
        return env
    if os.path.isfile('./voxelyze'):
        return os.path.abspath('./voxelyze')
    found = shutil.which('voxelyze')
    if found:
        return found
    return './voxelyze'


# ── Simulator runner ──────────────────────────────────────────────────────────

def run_voxcraft(
    genome: np.ndarray,
    timeout: int = 120,
    sim_time: float = 1.0,
) -> Tuple[float, float, float]:
    """Run voxelyze on one genome.

    Args:
        genome:   (8,8,8) int array
        timeout:  subprocess timeout in seconds
        sim_time: simulation duration in seconds
    Returns:
        (dx, dy, dz) where dx = NormFinalDist fitness, dy = 0.0, dz = 0.0.
        Returns (0.0, 0.0, 0.0) on any failure.
    """
    binary = _find_binary()

    with tempfile.TemporaryDirectory() as tmpdir:
        vxa_path = os.path.join(tmpdir, 'robot.vxa')
        out_path = os.path.join(tmpdir, 'output.xml')

        xml = _serializer_genome_to_vxa(genome, sim_time=sim_time, fitness_file=out_path)
        with open(vxa_path, 'w') as f:
            f.write(xml)

        try:
            subprocess.run(
                [binary, '-f', vxa_path],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return (0.0, 0.0, 0.0)

        if not os.path.exists(out_path):
            return (0.0, 0.0, 0.0)

        fitness = _parse_fitness(out_path)
        return (fitness, 0.0, 0.0)


# ── DEAP fitness wrapper ──────────────────────────────────────────────────────

def evaluate_genome_m3(genome: np.ndarray, sim_time: float = 1.0) -> Tuple[float]:
    """Evaluate genome; apply LCC; return DEAP fitness tuple.

    Args:
        genome:   (8,8,8) int array
        sim_time: simulation duration in seconds
    Returns:
        (fitness,) — returns (0.0,) for invalid genomes
    """
    g = largest_connected_component(np.asarray(genome, dtype=int))
    if int(np.sum(g > 0)) < 8 or int(np.sum(g >= 2)) == 0:
        return (0.0,)
    dx, _, _ = run_voxcraft(g, sim_time=sim_time)
    return (dx,)


# ── Parallel batch eval ───────────────────────────────────────────────────────

def _eval_worker(args: tuple) -> Tuple[float]:
    """Top-level pool worker: receives (plain_array, sim_time)."""
    genome, sim_time = args
    return evaluate_genome_m3(np.asarray(genome, dtype=int), sim_time)


def batch_evaluate_parallel(
    genomes: List[np.ndarray],
    sim_time: float = 1.0,
    n_workers: Optional[int] = None,
) -> List[Tuple[float]]:
    """Evaluate genomes in parallel using multiprocessing.Pool.

    Args:
        genomes:   list of (8,8,8) numpy arrays
        sim_time:  simulation duration per genome
        n_workers: number of parallel processes (default os.cpu_count())
    Returns:
        list of (fitness,) tuples, same order as input
    """
    if n_workers is None:
        n_workers = os.cpu_count()
    args = [(np.asarray(g, dtype=int), sim_time) for g in genomes]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_eval_worker, args)
    return results


if __name__ == '__main__':
    print('Testing voxcraft_runner...')
    g = np.zeros((8, 8, 8), dtype=int)
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(0, 4):
                g[x, y, z] = 2 if (x + y + z) % 2 == 0 else 3
    print('evaluate_genome_m3:', evaluate_genome_m3(g))
    print('batch (n=2):', batch_evaluate_parallel([g, g], n_workers=2))
