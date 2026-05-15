"""tests/test_map_elites.py — Unit tests for src/milestone4/map_elites.py."""

import numpy as np
import pytest


def _valid_genome() -> np.ndarray:
    g = np.zeros((8, 8, 8), dtype=int)
    g[2:6, 2:6, 0:4] = 2
    g[2:6, 2:6, 0:2] = 3
    return g


# ── compute_behaviour ────────────────────────────────────────────────────────

def test_compute_behaviour_returns_valid_bins():
    """compute_behaviour returns a 2-tuple of ints in [0, 9]."""
    from src.milestone4.map_elites import compute_behaviour
    g = _valid_genome()
    bins = compute_behaviour(g, fitness=0.5)
    assert isinstance(bins, tuple), 'Must return a tuple'
    assert len(bins) == 2, 'Must return a 2-tuple'
    assert 0 <= bins[0] <= 9, f'Distance bin out of range: {bins[0]}'
    assert 0 <= bins[1] <= 9, f'Symmetry bin out of range: {bins[1]}'


def test_compute_behaviour_zero_fitness():
    """Fitness of 0 maps to distance bin 0."""
    from src.milestone4.map_elites import compute_behaviour
    g = _valid_genome()
    bins = compute_behaviour(g, fitness=0.0)
    assert bins[0] == 0


def test_compute_behaviour_max_fitness():
    """Fitness ≥ 1.0 maps to distance bin 9."""
    from src.milestone4.map_elites import compute_behaviour
    g = _valid_genome()
    bins = compute_behaviour(g, fitness=1.0)
    assert bins[0] == 9


def test_compute_behaviour_symmetry_range():
    """Symmetry bin is always in [0, 9] for any genome."""
    from src.milestone4.map_elites import compute_behaviour
    rng = np.random.default_rng(0)
    for _ in range(10):
        g = rng.integers(0, 4, size=(8, 8, 8))
        bins = compute_behaviour(g, fitness=0.3)
        assert 0 <= bins[1] <= 9


# ── run_map_elites ────────────────────────────────────────────────────────────

def test_map_elites_grid_fills_cells():
    """run_map_elites returns a dict and a list logbook."""
    from src.milestone4.map_elites import run_map_elites
    grid, log = run_map_elites(config={
        'n_iterations': 5,
        'batch_size':   3,
        'seed':         0,
        'save_dir':     'results/m4/test_me/',
    })
    assert isinstance(grid, dict), 'grid must be a dict'
    assert isinstance(log, list),  'logbook must be a list'
    assert len(log) > 0,           'logbook must have at least one entry'


def test_map_elites_cells_are_valid():
    """All grid cells have valid bin indices and positive fitness."""
    from src.milestone4.map_elites import run_map_elites
    grid, _ = run_map_elites(config={
        'n_iterations': 5,
        'batch_size':   4,
        'seed':         1,
        'save_dir':     'results/m4/test_me2/',
    })
    for (d_bin, s_bin), (genome, fitness) in grid.items():
        assert 0 <= d_bin <= 9, f'Invalid distance bin: {d_bin}'
        assert 0 <= s_bin <= 9, f'Invalid symmetry bin: {s_bin}'
        assert isinstance(genome, np.ndarray), 'Genome must be numpy array'
        assert genome.shape == (8, 8, 8), 'Genome must be (8,8,8)'
        assert fitness >= 0.0, f'Fitness must be non-negative, got {fitness}'


def test_map_elites_logbook_has_required_keys():
    """Logbook entries contain required stat keys."""
    from src.milestone4.map_elites import run_map_elites
    _, log = run_map_elites(config={
        'n_iterations': 3,
        'batch_size':   2,
        'seed':         2,
        'save_dir':     'results/m4/test_me3/',
    })
    required = {'iteration', 'n_filled_cells', 'max_fitness', 'mean_fitness'}
    for rec in log:
        assert required.issubset(rec.keys()), f'Missing keys in log record: {rec.keys()}'


# ── plot_map_elites_grid ──────────────────────────────────────────────────────

def test_plot_map_elites_grid_empty():
    """plot_map_elites_grid handles an empty grid without crashing."""
    from src.milestone4.map_elites import plot_map_elites_grid
    import os
    plot_map_elites_grid({}, save_path='results/m4/test_grid_empty.png')
    assert os.path.exists('results/m4/test_grid_empty.png')
