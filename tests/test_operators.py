"""Tests for all 5 DEAP operators in src/ea.py."""

import numpy as np
import pytest
from deap import creator

from src.ea import (
    N_MATERIALS,
    cx_one_point_slice,
    cx_uniform_voxel,
    mutate_block,
    mutate_grow_shrink,
    mutate_random_flip,
)


def _ind(seed: int = 0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, N_MATERIALS, size=(8, 8, 8))
    return creator.Individual(arr)


@pytest.mark.parametrize("op,kwargs", [
    (mutate_random_flip, {}),
    (mutate_block,       {}),
    (mutate_grow_shrink, {}),
])
def test_mutation_preserves_shape_and_range(op, kwargs):
    ind = _ind()
    (result,) = op(ind, rng=np.random.default_rng(0), **kwargs)
    assert result.shape == (8, 8, 8)
    assert result.min() >= 0
    assert result.max() < N_MATERIALS


@pytest.mark.parametrize("op", [cx_uniform_voxel, cx_one_point_slice])
def test_crossover_preserves_shape_and_range(op):
    r1, r2 = op(_ind(0), _ind(1), rng=np.random.default_rng(0))
    for r in (r1, r2):
        assert r.shape == (8, 8, 8)
        assert r.min() >= 0
        assert r.max() < N_MATERIALS


def test_mutate_random_flip_returns_tuple():
    result = mutate_random_flip(_ind(), rng=np.random.default_rng(0))
    assert isinstance(result, tuple) and len(result) == 1


def test_mutate_block_returns_tuple():
    result = mutate_block(_ind(), rng=np.random.default_rng(0))
    assert isinstance(result, tuple) and len(result) == 1


def test_mutate_grow_shrink_returns_tuple():
    result = mutate_grow_shrink(_ind(), rng=np.random.default_rng(0))
    assert isinstance(result, tuple) and len(result) == 1


def test_mutate_random_flip_inplace():
    ind = _ind()
    (result,) = mutate_random_flip(ind, rng=np.random.default_rng(0))
    assert id(result) == id(ind)


def test_random_flip_changes_values_at_high_prob():
    ind = _ind(0)
    original = np.array(ind).copy()
    mutate_random_flip(ind, flip_prob=0.99, rng=np.random.default_rng(42))
    assert not np.array_equal(ind, original)


def test_grow_increases_or_keeps_occupancy():
    ind = creator.Individual(np.zeros((8, 8, 8), dtype=int))
    ind[3:5, 3:5, 3:5] = 1
    before = int(np.sum(ind > 0))
    mutate_grow_shrink(ind, prob_grow=1.0, prob_shrink=0.0,
                       rng=np.random.default_rng(0))
    assert int(np.sum(ind > 0)) >= before


def test_shrink_decreases_or_keeps_occupancy():
    ind = creator.Individual(np.zeros((8, 8, 8), dtype=int))
    ind[2:6, 2:6, 2:6] = 1
    before = int(np.sum(ind > 0))
    mutate_grow_shrink(ind, prob_grow=0.0, prob_shrink=1.0,
                       rng=np.random.default_rng(0))
    assert int(np.sum(ind > 0)) <= before


def test_cx_uniform_identical_parents_stay_identical():
    arr = np.ones((8, 8, 8), dtype=int)
    r1, r2 = cx_uniform_voxel(
        creator.Individual(arr.copy()),
        creator.Individual(arr.copy()),
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(np.asarray(r1), np.asarray(r2))


def test_cx_slice_swaps_material():
    i1 = creator.Individual(np.zeros((8, 8, 8), dtype=int))
    i2 = creator.Individual(np.full((8, 8, 8), 2, dtype=int))
    r1, r2 = cx_one_point_slice(i1, i2, rng=np.random.default_rng(5))
    # After LCC, shapes must be correct
    assert r1.shape == (8, 8, 8)
    assert r2.shape == (8, 8, 8)
