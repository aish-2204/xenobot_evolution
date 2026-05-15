"""src/milestone3/voxcraft_runner.py

Unified voxelyze / voxcraft-sim subprocess runner.

Auto-detects binary from VOXCRAFT_BIN basename:
  voxelyze     (arm64 macOS, local dev)
      voxelyze -f robot.vxa
      output path embedded in VXA <GA><FitnessFileName>

  voxcraft-sim (CUDA Linux, Colab GPU runtime)
      voxcraft-sim -i input_dir/ -o output.xml -l -f
      must be run from the binary's own directory (build/)

genome_to_vxa() output is identical for both — no format changes needed.
"""

import os
import subprocess
import tempfile
from multiprocessing.pool import ThreadPool
from typing import Callable, List, Optional

import numpy as np

from src.milestone1.fitness import _parse_fitness, get_bin
from src.milestone1.genome import largest_connected_component
from src.milestone1.serializer import genome_to_vxa


# ── Binary detection ─────────────────────────────────────────────────────────

def _binary_type(path: str) -> str:
    """Return 'voxcraft-sim' or 'voxelyze' based on the binary filename."""
    return "voxcraft-sim" if "voxcraft" in os.path.basename(path).lower() else "voxelyze"


# ── Per-binary runners ───────────────────────────────────────────────────────

def _run_voxelyze(genome: np.ndarray, binary: str, sim_time: float):
    """voxelyze -f robot.vxa   (output path embedded in VXA)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vxa_path = os.path.join(tmpdir, "robot.vxa")
        out_path = os.path.join(tmpdir, "output.xml")

        with open(vxa_path, "w") as f:
            f.write(genome_to_vxa(genome, sim_time=sim_time, fitness_file=out_path))

        try:
            proc = subprocess.run(
                [binary, "-f", vxa_path],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"

        if not os.path.exists(out_path):
            return 0.0, proc.stderr[-300:]
        return _parse_fitness(out_path), None


def _run_voxcraft_sim(genome: np.ndarray, binary: str, sim_time: float):
    """voxcraft-sim -i input_dir/ -o output.xml -l -f

    Must run from the binary's own directory (build/).
    input_dir contains robot.vxa; output.xml receives the fitness result.
    """
    build_dir = os.path.dirname(os.path.abspath(binary))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)

        vxa_path = os.path.join(input_dir, "robot.vxa")
        out_path = os.path.join(tmpdir, "output.xml")

        with open(vxa_path, "w") as f:
            f.write(genome_to_vxa(genome, sim_time=sim_time, fitness_file=out_path))

        try:
            proc = subprocess.run(
                [binary, "-i", input_dir + "/", "-o", out_path, "-l", "-f"],
                capture_output=True, text=True, timeout=120,
                cwd=build_dir,
            )
        except subprocess.TimeoutExpired:
            return 0.0, "timeout"

        if not os.path.exists(out_path):
            return 0.0, proc.stderr[-300:]
        return _parse_fitness(out_path), None


# ── Public API ───────────────────────────────────────────────────────────────

def run_voxcraft(
    genome: np.ndarray,
    bin_path: Optional[str] = None,
    sim_time: float = 0.5,
    apply_lcc: bool = True,
    raise_on_failure: bool = False,
) -> float:
    """Evaluate one genome. Auto-detects voxelyze vs voxcraft-sim from binary name.

    Args:
        genome:           (8,8,8) int numpy array (values 0-3)
        bin_path:         explicit binary path; None → VOXCRAFT_BIN env var
        sim_time:         simulation duration in seconds
        apply_lcc:        apply LCC before simulation (default True)
        raise_on_failure: raise RuntimeError if sim produces no output

    Returns:
        float >= 0.0  (NormFinalDist)
    """
    binary = bin_path or get_bin()

    g = largest_connected_component(np.asarray(genome, dtype=int)) if apply_lcc \
        else np.asarray(genome, dtype=int)

    if int(np.sum(g > 0)) < 8 or int(np.sum(g >= 2)) == 0:
        return 0.0

    if _binary_type(binary) == "voxcraft-sim":
        score, err = _run_voxcraft_sim(g, binary, sim_time)
    else:
        score, err = _run_voxelyze(g, binary, sim_time)

    if err and raise_on_failure:
        raise RuntimeError(
            f"{os.path.basename(binary)} produced no output.\n"
            f"  binary: {binary}\n"
            f"  stderr: {err}"
        )
    return score


def _thread_worker(args):
    genome, bin_path, sim_time, apply_lcc = args
    return run_voxcraft(genome, bin_path=bin_path, sim_time=sim_time, apply_lcc=apply_lcc)


def batch_evaluate_m3(
    genomes: List[np.ndarray],
    n_workers: int = 4,
    sim_time: float = 0.5,
    bin_path: Optional[str] = None,
    apply_lcc: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    use_process_pool: bool = False,
) -> List[float]:
    """Evaluate a list of genomes in parallel.

    Args:
        genomes:          list of (8,8,8) numpy arrays
        n_workers:        parallel workers (Colab free ~2 CPUs, Pro ~8)
        sim_time:         simulation duration per genome
        bin_path:         explicit binary path (None → env var)
        apply_lcc:        apply LCC before each evaluation
        progress_cb:      optional callable(done, total) called after each result
        use_process_pool: ProcessPool instead of ThreadPool (Linux/Colab only)

    Returns:
        list of float fitness scores (same order as input)
    """
    if not genomes:
        return []

    resolved_bin = bin_path or get_bin()
    args = [(g, resolved_bin, sim_time, apply_lcc) for g in genomes]

    if use_process_pool:
        from multiprocessing import Pool
        PoolClass = Pool
    else:
        PoolClass = ThreadPool

    scores: List[float] = []
    with PoolClass(n_workers) as pool:
        if progress_cb is None:
            scores = list(pool.map(_thread_worker, args))
        else:
            for i, score in enumerate(pool.imap(_thread_worker, args)):
                scores.append(score)
                progress_cb(i + 1, len(genomes))
    return scores


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    binary = get_bin()
    btype  = _binary_type(binary)
    print(f"Binary : {binary}")
    print(f"Type   : {btype}")

    g = np.zeros((8, 8, 8), dtype=int)
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(0, 4):
                g[x, y, z] = 2 if (x + y + z) % 2 == 0 else 3

    print(f"Testing 4x4x4 alternating-active genome ({int(np.sum(g > 0))} voxels)...")
    score = run_voxcraft(g, sim_time=0.5)
    print(f"Score  : {score:.6f}")
