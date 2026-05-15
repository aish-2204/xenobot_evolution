import os
import tempfile
import numpy as np
import pytest
from xml.etree import ElementTree as ET

from src.milestone1.serializer import genome_to_vxa, vxa_to_file


# ── helpers ───────────────────────────────────────────────────────────────────

def _sparse_genome() -> np.ndarray:
    """4×4×4 block of passive voxels surrounded by empty space."""
    g = np.zeros((8, 8, 8), dtype=int)
    g[2:6, 2:6, 0:4] = 1
    return g


def _active_genome() -> np.ndarray:
    """Alternating active+/active- with some passive and empty voxels."""
    g = np.zeros((8, 8, 8), dtype=int)
    for x in range(2, 6):
        for y in range(2, 6):
            for z in range(0, 4):
                g[x, y, z] = 2 if (x + y + z) % 2 == 0 else 3
    g[0, 0, 0] = 1  # one passive voxel
    return g


# ── test 1: valid XML ─────────────────────────────────────────────────────────

def test_output_is_valid_xml():
    g = _active_genome()
    xml_str = genome_to_vxa(g)
    # fromstring raises if the XML is malformed
    ET.fromstring(xml_str)


def test_xml_has_required_root_and_sections():
    xml_str = genome_to_vxa(_active_genome())
    root = ET.fromstring(xml_str)
    assert root.tag == "VXA"
    assert root.find("Simulator") is not None
    assert root.find("Environment") is not None
    assert root.find(".//Structure") is not None
    assert root.find(".//Palette") is not None


# ── test 2: no PhaseOffset tag ────────────────────────────────────────────────

def test_no_phase_offset_tag_anywhere():
    """active- uses negative CTE — PhaseOffset must not appear in output."""
    xml_str = genome_to_vxa(_active_genome())
    assert "PhaseOffset" not in xml_str


def test_passive_material_has_zero_cte():
    xml_str = genome_to_vxa(_sparse_genome())
    root = ET.fromstring(xml_str)
    passive = root.find(".//Material[@ID='1']")
    assert passive is not None
    cte = passive.find(".//CTE")
    assert cte is not None
    assert float(cte.text.strip()) == 0.0


def test_active_minus_has_negative_cte():
    xml_str = genome_to_vxa(_active_genome())
    root = ET.fromstring(xml_str)
    mat3 = root.find(".//Material[@ID='3']")
    assert mat3 is not None
    cte = mat3.find(".//CTE")
    assert cte is not None
    assert float(cte.text.strip()) < 0.0


# ── test 3: empty voxels not in material palette ──────────────────────────────

def test_no_material_id_zero_in_palette():
    """Material ID=0 must not exist — empty voxels are represented as '0' in data."""
    xml_str = genome_to_vxa(_sparse_genome())
    root = ET.fromstring(xml_str)
    assert root.find(".//Material[@ID='0']") is None


def test_empty_genome_still_valid_xml():
    g = np.zeros((8, 8, 8), dtype=int)
    xml_str = genome_to_vxa(g)
    ET.fromstring(xml_str)  # must not raise


def test_data_layer_count_equals_z_dim():
    g = _active_genome()
    xml_str = genome_to_vxa(g)
    root = ET.fromstring(xml_str)
    layers = root.findall(".//Layer")
    assert len(layers) == g.shape[2]


def test_data_layer_length_equals_x_times_y():
    g = _active_genome()
    sx, sy, _ = g.shape
    xml_str = genome_to_vxa(g)
    root = ET.fromstring(xml_str)
    for layer in root.findall(".//Layer"):
        assert len(layer.text.strip()) == sx * sy


# ── test 4: sim_time and fitness_file propagate ───────────────────────────────

def test_sim_time_in_output():
    xml_str = genome_to_vxa(_active_genome(), sim_time=2.5)
    root = ET.fromstring(xml_str)
    val = root.find(".//StopConditionValue")
    assert val is not None
    assert float(val.text.strip()) == pytest.approx(2.5)


def test_fitness_file_in_output():
    xml_str = genome_to_vxa(_active_genome(), fitness_file="/tmp/my_result.xml")
    assert "/tmp/my_result.xml" in xml_str


# ── test 5: vxa_to_file writes to disk ───────────────────────────────────────

def test_vxa_to_file_creates_file():
    g = _active_genome()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "robot.vxa")
        vxa_to_file(g, path)
        assert os.path.exists(path)
        content = open(path).read()
        ET.fromstring(content)  # still valid XML


def test_vxa_to_file_creates_parent_dirs():
    g = _active_genome()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "subdir", "robot.vxa")
        vxa_to_file(g, path)
        assert os.path.exists(path)
