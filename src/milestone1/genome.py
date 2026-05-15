"""Genome representation for 8×8×8 voxel soft robots."""

import numpy as np
from scipy import ndimage

EMPTY = 0
PASSIVE = 1
ACTIVE_P = 2
ACTIVE_N = 3

MATERIAL_NAMES = {0: "empty", 1: "passive", 2: "active_plus", 3: "active_minus"}


def random_genome(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random 8×8×8 genome with voxel values in {0, 1, 2, 3}.

    Args:
        rng: Optional seeded random number generator.
    Returns:
        Integer array of shape (8, 8, 8).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 4, size=(8, 8, 8))


def largest_connected_component(genome: np.ndarray) -> np.ndarray:
    """Keep only the largest 6-connected body of non-empty voxels.

    Args:
        genome: Integer array of shape (8, 8, 8), values 0-3.
    Returns:
        Copy of genome with voxels outside the largest component zeroed out.
    """
    result = np.copy(genome)
    occupied = genome > 0
    if not occupied.any():
        return result

    # scipy default structure gives 6-connectivity in 3D
    labeled, n_components = ndimage.label(occupied)
    if n_components <= 1:
        return result

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # exclude background label
    largest_label = sizes.argmax()
    result[labeled != largest_label] = 0
    return result


def voxel_counts(genome: np.ndarray) -> dict[str, int]:
    """Count voxels by material type.

    Args:
        genome: Integer array of shape (8, 8, 8).
    Returns:
        Dict with keys 'empty', 'passive', 'active_plus', 'active_minus'.
    """
    return {name: int(np.sum(genome == val)) for val, name in MATERIAL_NAMES.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    g = random_genome(rng)
    g_lcc = largest_connected_component(g)
    counts_before = voxel_counts(g)
    counts_after = voxel_counts(g_lcc)
    removed = sum(counts_before[k] for k in ("passive", "active_plus", "active_minus")) - \
              sum(counts_after[k] for k in ("passive", "active_plus", "active_minus"))
    print(f"Shape: {g.shape}, dtype: {g.dtype}")
    print(f"Before LCC: {counts_before}")
    print(f"After LCC:  {counts_after}")
    print(f"Isolated voxels removed: {removed}")
