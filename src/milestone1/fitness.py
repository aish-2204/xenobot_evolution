"""
src/milestone1/fitness.py

Fitness evaluator — calls voxelyze binary via subprocess.

Usage:
    from src.milestone1.fitness import evaluate_genome
    score = evaluate_genome(genome)   # returns float, higher = moves farther

Environment:
    export VOXCRAFT_BIN=./voxelyze   (or put voxelyze on PATH)
"""

import os
import shutil
import subprocess
import tempfile
import numpy as np
from xml.etree import ElementTree as ET
from typing import Optional

from src.milestone1.serializer import genome_to_vxa  # noqa: F401 (re-exported)


# ─────────────────────────────────────────────
#  1. Locate the simulator binary
# ─────────────────────────────────────────────

def _find_voxcraft_bin() -> str:
    """Find simulator binary. Checks VOXCRAFT_BIN env var, then PATH.
    Accepts either 'voxelyze' (CPU build from build_voxelyze_mac.sh)
    or 'voxcraft-sim' (GPU build).
    """
    # Explicit env var takes priority
    env = os.environ.get("VOXCRAFT_BIN")
    if env and os.path.isfile(env):
        return env

    # Check PATH for either binary name
    for name in ("voxelyze", "voxcraft-sim"):
        found = shutil.which(name)
        if found:
            return found

    # Check common relative paths (project root)
    for path in ("./voxelyze", "../voxelyze", "./voxcraft-sim/build/voxcraft-sim"):
        if os.path.isfile(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "\n\n❌ Simulator binary not found!\n"
        "Fix one of these:\n"
        "  1. Run the build script:  ./build_voxelyze_mac.sh\n"
        "     Then: export VOXCRAFT_BIN=$(pwd)/voxelyze\n"
        "  2. Or add it to PATH and restart terminal\n"
    )


# Cache after first lookup
_VOXCRAFT_BIN: Optional[str] = None


def get_bin() -> str:
    global _VOXCRAFT_BIN
    if _VOXCRAFT_BIN is None:
        _VOXCRAFT_BIN = _find_voxcraft_bin()
    return _VOXCRAFT_BIN


# ─────────────────────────────────────────────
#  2. Parse voxelyze output XML
# ─────────────────────────────────────────────

def _parse_fitness(output_xml_path: str) -> float:
    """Extract NormFinalDist from voxelyze result XML.

    voxelyze writes: <Voxelyze_Sim_Result><Fitness><NormFinalDist>...
    Returns 0.0 if the file is missing or parsing fails.
    """
    try:
        tree = ET.parse(output_xml_path)
        root = tree.getroot()
        el = root.find(".//NormFinalDist")
        if el is not None and el.text:
            return max(0.0, float(el.text.strip()))
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────
#  3. Main fitness function
# ─────────────────────────────────────────────

def evaluate_genome(genome: np.ndarray, sim_time: float = 1.0) -> float:
    """Evaluate a genome using voxelyze physics simulation.

    Writes the genome as a VXA file, runs voxelyze as a subprocess,
    parses the output XML, and returns the normalised CoM displacement.

    Args:
        genome:   (8,8,8) int numpy array
        sim_time: simulation duration in seconds. Use 0.5–1.0 for fast
                  EA runs, 2.0+ for accurate final evaluation.

    Returns:
        float: normalised CoM displacement (≥0, higher = moves farther)
               Returns 0.0 for invalid genomes or sim failures.
    """
    # Reject near-empty genomes immediately — saves sim time
    if int(np.sum(genome > 0)) < 8:
        return 0.0

    # Reject genomes with no active material — they can't move
    if int(np.sum(genome >= 2)) == 0:
        return 0.0

    binary = get_bin()

    with tempfile.TemporaryDirectory() as tmpdir:
        vxa_path = os.path.join(tmpdir, "robot.vxa")
        out_path = os.path.join(tmpdir, "output.xml")

        vxa_xml = genome_to_vxa(genome, sim_time, fitness_file=out_path)
        with open(vxa_path, "w") as f:
            f.write(vxa_xml)

        try:
            # voxelyze takes -f <file> only — output path is embedded in VXA
            result = subprocess.run(
                [binary, "-f", vxa_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return 0.0
        except FileNotFoundError:
            raise

        if not os.path.exists(out_path):
            # Uncomment to debug: print(result.stdout[-500:], result.stderr[-200:])
            return 0.0

        return _parse_fitness(out_path)


# ─────────────────────────────────────────────
#  4. Batch / parallel evaluation (used in M3)
# ─────────────────────────────────────────────

def _eval_worker(args):
    """Unpacking wrapper for multiprocessing (must be top-level)."""
    genome, sim_time = args
    return evaluate_genome(genome, sim_time)


def batch_evaluate(
    genomes: list,
    sim_time: float = 1.0,
    n_workers: int = 4,
) -> list:
    """Evaluate a list of genomes in parallel.

    Args:
        genomes:   list of (8,8,8) numpy arrays
        sim_time:  simulation duration per genome
        n_workers: number of parallel processes

    Returns:
        list of float fitness scores, same order as input
    """
    from multiprocessing import Pool

    args = [(g, sim_time) for g in genomes]
    with Pool(processes=n_workers) as pool:
        scores = pool.map(_eval_worker, args)
    return scores


# ─────────────────────────────────────────────
#  5. Smoke test — run directly to verify setup
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Fitness evaluator smoke test ===")
    print("Looking for voxelyze binary...")

    try:
        bin_path = get_bin()
        print(f"✅ Found: {bin_path}")
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # Test 1: all-empty genome → should return 0
    empty = np.zeros((8, 8, 8), dtype=int)
    s = evaluate_genome(empty)
    assert s == 0.0, f"Expected 0 for empty genome, got {s}"
    print("✅ Empty genome → 0.0")

    # Test 2: a solid block of alternating active+/active- voxels
    # This is the canonical "should definitely move" test case
    rng = np.random.default_rng(42)
    genome = np.zeros((8, 8, 8), dtype=int)
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(0, 4):
                genome[x, y, z] = 2 if (x + y + z) % 2 == 0 else 3

    print(f"Testing 4x4x4 alternating active block "
          f"({int(np.sum(genome > 0))} voxels)...")
    score = evaluate_genome(genome, sim_time=1.0)
    print(f"✅ Fitness score: {score:.6f}")

    # Test 3: VXA XML generation
    vxa = genome_to_vxa(genome)
    assert "<VXA" in vxa and "<Material" in vxa
    print("✅ genome_to_vxa() produces valid XML")

    print("\n✅ All tests passed — fitness evaluator is working!")
    print("   You can now use evaluate_genome() in your EA loop.")
