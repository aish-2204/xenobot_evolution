"""tests/test_multiobjective.py — Unit tests for src/multiobjective.py."""

import numpy as np
import pytest
from deap import base, creator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mo_population(fitness_values: list) -> list:
    """Build a list of IndividualMO with preset fitness values."""
    from src.multiobjective import creator as _c  # noqa: ensures types registered
    from src import multiobjective  # noqa: triggers type creation

    from deap import creator as dc
    g = np.zeros((8, 8, 8), dtype=int)
    pop = []
    for vals in fitness_values:
        ind = dc.IndividualMO(g.copy())
        ind.fitness.values = vals
        pop.append(ind)
    return pop


def _valid_genome() -> np.ndarray:
    """Return a genome that should produce a non-zero evaluation."""
    g = np.zeros((8, 8, 8), dtype=int)
    g[2:6, 2:6, 0:4] = 2
    g[2:6, 2:6, 0:2] = 3
    return g


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_evaluate_mo_returns_tuple():
    """evaluate_mo returns a 2-tuple of floats."""
    from src.multiobjective import evaluate_mo
    g = _valid_genome()
    result = evaluate_mo(g)
    assert isinstance(result, tuple), 'Must return a tuple'
    assert len(result) == 2, 'Must return a 2-tuple'
    assert isinstance(result[0], float), 'First element (distance) must be float'
    assert isinstance(result[1], float), 'Second element (n_voxels) must be float'


def test_evaluate_mo_empty_genome():
    """evaluate_mo returns (0.0, 0.0) for an all-empty genome."""
    from src.multiobjective import evaluate_mo
    g = np.zeros((8, 8, 8), dtype=int)
    result = evaluate_mo(g)
    assert result == (0.0, 0.0), f'Expected (0.0, 0.0) for empty genome, got {result}'


def test_evaluate_mo_voxel_count():
    """n_voxels in evaluate_mo output equals number of non-empty voxels after LCC."""
    from src.multiobjective import evaluate_mo
    from src.milestone1.genome import largest_connected_component, EMPTY
    g = np.zeros((8, 8, 8), dtype=int)
    g[3:7, 3:7, 0:4] = 2
    g[3:7, 3:7, 0:2] = 3
    g_lcc = largest_connected_component(g)
    expected_voxels = float((g_lcc != EMPTY).sum())
    _, n_voxels = evaluate_mo(g)
    assert n_voxels == expected_voxels, f'Expected {expected_voxels} voxels, got {n_voxels}'


def test_extract_pareto_front():
    """extract_pareto_front returns a subset of the input population."""
    import src.multiobjective  # noqa: triggers creator registration
    from src.multiobjective import extract_pareto_front
    from deap import creator as dc

    pop = _make_mo_population([(0.8, 10.0), (0.3, 5.0), (0.6, 8.0), (0.5, 15.0)])
    front = extract_pareto_front(pop)

    assert isinstance(front, list), 'Pareto front must be a list'
    assert 0 < len(front) <= len(pop), 'Front must be a non-empty subset'
    pop_ids = {id(x) for x in pop}
    for ind in front:
        assert id(ind) in pop_ids, 'All front members must come from the population'


def test_extract_pareto_front_empty():
    """extract_pareto_front returns [] for empty population."""
    from src.multiobjective import extract_pareto_front
    assert extract_pareto_front([]) == []


def test_find_knee_point():
    """find_knee_point returns a single individual from the population."""
    import src.multiobjective  # noqa
    from src.multiobjective import find_knee_point

    pop = _make_mo_population([(0.5, 10.0), (0.8, 30.0), (1.0, 50.0)])
    knee = find_knee_point(pop)
    pop_ids = {id(x) for x in pop}
    assert knee is not None, 'Knee point must not be None for non-empty front'
    assert id(knee) in pop_ids, 'Knee point must be a member of the input population'


def test_find_knee_point_empty():
    """find_knee_point returns None for empty front."""
    from src.multiobjective import find_knee_point
    assert find_knee_point([]) is None


def test_build_toolbox_mo():
    """build_toolbox_mo returns a Toolbox with required operators."""
    from src.multiobjective import build_toolbox_mo
    tb = build_toolbox_mo()
    for attr in ('individual', 'population', 'evaluate', 'mate', 'mutate', 'select'):
        assert hasattr(tb, attr), f'Toolbox missing operator: {attr}'


def test_compute_hypervolume_returns_float():
    """compute_hypervolume returns a non-negative float."""
    import src.multiobjective  # noqa
    from src.multiobjective import compute_hypervolume
    pop = _make_mo_population([(0.5, 10.0), (0.3, 5.0)])
    hv = compute_hypervolume(pop)
    assert isinstance(hv, float)
    assert hv >= 0.0
