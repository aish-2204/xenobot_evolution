"""tests/test_voxcraft_runner.py — Unit tests for src/voxcraft_runner.py"""

import os
import tempfile
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from src.voxcraft_runner import (
    batch_evaluate_parallel,
    evaluate_genome_m3,
    genome_to_vxa,
    run_voxcraft,
)


def _active_genome() -> np.ndarray:
    """4×4×4 alternating active+/active− block — known to move."""
    g = np.zeros((8, 8, 8), dtype=int)
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(0, 4):
                g[x, y, z] = 2 if (x + y + z) % 2 == 0 else 3
    return g


# ── TASK 1 tests ──────────────────────────────────────────────────────────────

def test_genome_to_vxa_creates_file():
    """genome_to_vxa must write a file that parses as valid XML."""
    g = _active_genome()
    with tempfile.TemporaryDirectory() as tmpdir:
        vxa_path = os.path.join(tmpdir, 'robot.vxa')
        genome_to_vxa(g, vxa_path, sim_time=0.5)

        assert os.path.isfile(vxa_path), 'VXA file was not created'

        tree = ET.parse(vxa_path)
        root = tree.getroot()
        assert root.tag == 'VXA', f'Expected root tag VXA, got {root.tag}'

        # Check key structural elements are present
        assert root.find('.//Simulator') is not None
        assert root.find('.//Structure') is not None
        assert root.find('.//FitnessFileName') is not None


def test_run_voxcraft_returns_tuple():
    """run_voxcraft must return a 3-tuple of floats."""
    g = _active_genome()
    result = run_voxcraft(g, sim_time=0.5)

    assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
    assert len(result) == 3, f'Expected 3-tuple, got length {len(result)}'
    for i, val in enumerate(result):
        assert isinstance(val, float), f'Element {i} is not float: {type(val)}'


def test_evaluate_genome_m3_empty():
    """Empty genome must return (0.0,)."""
    empty = np.zeros((8, 8, 8), dtype=int)
    result = evaluate_genome_m3(empty)

    assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
    assert len(result) == 1, f'Expected 1-tuple, got {result}'
    assert result[0] == 0.0, f'Expected 0.0 for empty genome, got {result[0]}'


def test_evaluate_genome_m3_active():
    """Active genome must return positive fitness."""
    g = _active_genome()
    result = evaluate_genome_m3(g, sim_time=0.5)

    assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
    assert len(result) == 1, f'Expected 1-tuple, got {result}'
    assert result[0] > 0.0, (
        f'Expected positive fitness for active genome, got {result[0]}. '
        'Check that VOXCRAFT_BIN is set and voxelyze is working.'
    )
