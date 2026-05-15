import numpy as np
import pytest
from src.milestone1.genome import random_genome, largest_connected_component, voxel_counts


def test_shape():
    g = random_genome()
    assert g.shape == (8, 8, 8)


def test_value_range():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert g.min() >= 0
    assert g.max() <= 3


def test_dtype_is_integer():
    g = random_genome()
    assert np.issubdtype(g.dtype, np.integer)


def test_lcc_removes_isolated_voxels():
    g = np.zeros((8, 8, 8), dtype=int)
    g[0:3, 0:3, 0:3] = 1   # large 3×3×3 block
    g[7, 7, 7] = 2          # isolated single voxel
    g_lcc = largest_connected_component(g)
    assert g_lcc[7, 7, 7] == 0, "isolated voxel should be zeroed"
    assert g_lcc[1, 1, 1] == 1, "main block should survive"


def test_lcc_all_zero_genome():
    g = np.zeros((8, 8, 8), dtype=int)
    g_lcc = largest_connected_component(g)
    assert np.all(g_lcc == 0)


def test_lcc_preserves_shape():
    rng = np.random.default_rng(42)
    g = random_genome(rng)
    g_lcc = largest_connected_component(g)
    assert g_lcc.shape == (8, 8, 8)


def test_lcc_does_not_mutate_input():
    g = np.zeros((8, 8, 8), dtype=int)
    g[0:4, 0:4, 0:4] = 1
    g[7, 7, 7] = 1
    original = g.copy()
    largest_connected_component(g)
    assert np.array_equal(g, original)


def test_lcc_single_component_unchanged():
    g = np.zeros((8, 8, 8), dtype=int)
    g[2:6, 2:6, 2:6] = 1  # one solid block, no isolated pieces
    g_lcc = largest_connected_component(g)
    assert np.array_equal(g, g_lcc)


def test_voxel_counts_sum_to_total():
    rng = np.random.default_rng(7)
    g = random_genome(rng)
    counts = voxel_counts(g)
    assert sum(counts.values()) == 8 * 8 * 8


def test_voxel_counts_specific():
    g = np.zeros((8, 8, 8), dtype=int)
    g[0, 0, 0] = 1
    g[1, 1, 1] = 2
    g[2, 2, 2] = 3
    counts = voxel_counts(g)
    assert counts["passive"] == 1
    assert counts["active_plus"] == 1
    assert counts["active_minus"] == 1
    assert counts["empty"] == 8 * 8 * 8 - 3
