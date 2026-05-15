"""VXA serializer: convert genome arrays to voxelyze simulation files."""

import os
import numpy as np

# 0=empty (not in palette), 1=passive, 2=active+, 3=active-
# active- uses negative CTE (contracts on heat) — NOT PhaseOffset, which is
# per-voxel in voxelyze and not per-material.
_MATERIALS = {
    1: dict(name="Passive", cte="0"),
    2: dict(name="Active+", cte="0.01"),
    3: dict(name="Active-", cte="-0.01"),
}


def genome_to_vxa(
    genome: np.ndarray,
    sim_time: float = 1.0,
    fitness_file: str = "output.xml",
) -> str:
    """Convert an 8×8×8 genome to a VXA XML string for voxelyze.

    Args:
        genome:       (8,8,8) int array — 0=empty, 1=passive, 2=active+, 3=active-
        sim_time:     simulation duration in seconds
        fitness_file: path where voxelyze writes its result XML (set in <GA> block)

    Returns:
        Complete VXA XML string.
    """
    sx, sy, sz = genome.shape

    mat_xml = ""
    for mat_id, props in _MATERIALS.items():
        mat_xml += f"""
      <Material ID="{mat_id}">
        <MatType>0</MatType>
        <Name>{props['name']}</Name>
        <Mechanical>
          <MatModel>0</MatModel>
          <Elastic_Mod>1e+007</Elastic_Mod>
          <Plastic_Mod>0</Plastic_Mod>
          <Yield_Stress>0</Yield_Stress>
          <Density>1e+006</Density>
          <Poissons_Ratio>0.35</Poissons_Ratio>
          <CTE>{props['cte']}</CTE>
          <uStatic>1</uStatic>
          <uDynamic>0.5</uDynamic>
        </Mechanical>
      </Material>"""

    # Layers stacked in Z; each layer is row-major over X then Y
    layers_xml = ""
    for z in range(sz):
        row = "".join(str(int(genome[x, y, z])) for y in range(sy) for x in range(sx))
        layers_xml += f"\n      <Layer><![CDATA[{row}]]></Layer>"

    return f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<VXA Version="1.0">
  <Simulator>
    <Integration>
      <Integrator>0</Integrator>
      <DtFrac>0.9</DtFrac>
    </Integration>
    <Damping>
      <BondDampingZ>1</BondDampingZ>
      <ColDampingZ>0.8</ColDampingZ>
      <SlowDampingZ>1.7378e-005</SlowDampingZ>
    </Damping>
    <StopCondition>
      <StopConditionType>2</StopConditionType>
      <StopConditionValue>{sim_time}</StopConditionValue>
    </StopCondition>
    <GA>
      <FitnessFileName>{fitness_file}</FitnessFileName>
      <WriteFitnessFile>true</WriteFitnessFile>
    </GA>
  </Simulator>
  <Environment>
    <Fixed_Regions><NumFixed>0</NumFixed></Fixed_Regions>
    <Forced_Regions><NumForced>0</NumForced></Forced_Regions>
    <Gravity>
      <GravEnabled>1</GravEnabled>
      <GravAcc>-9.81</GravAcc>
      <FloorEnabled>1</FloorEnabled>
    </Gravity>
    <Thermal>
      <TempEnabled>1</TempEnabled>
      <TempAmp>39</TempAmp>
      <TempBase>25</TempBase>
      <VaryTempEnabled>1</VaryTempEnabled>
      <TempPeriod>0.025</TempPeriod>
    </Thermal>
  </Environment>
  <VXC Version="0.93">
    <Lattice>
      <Lattice_Dim>0.001</Lattice_Dim>
    </Lattice>
    <Voxel>
      <Vox_Name>BOX</Vox_Name>
      <X_Squeeze>1</X_Squeeze>
      <Y_Squeeze>1</Y_Squeeze>
      <Z_Squeeze>1</Z_Squeeze>
    </Voxel>
    <Palette>{mat_xml}
    </Palette>
    <Structure Compression="ASCII_READABLE">
      <X_Voxels>{sx}</X_Voxels>
      <Y_Voxels>{sy}</Y_Voxels>
      <Z_Voxels>{sz}</Z_Voxels>
      <Data>{layers_xml}
      </Data>
    </Structure>
  </VXC>
</VXA>"""


def vxa_to_file(genome: np.ndarray, path: str, sim_time: float = 1.0) -> None:
    """Write a genome as a VXA file to disk.

    The fitness output path is derived from the VXA path (same dir, _output.xml suffix).

    Args:
        genome:   (8,8,8) int array
        path:     destination file path (e.g. "results/m1/robot.vxa")
        sim_time: simulation duration in seconds
    """
    fitness_file = os.path.splitext(os.path.abspath(path))[0] + "_output.xml"
    xml = genome_to_vxa(genome, sim_time=sim_time, fitness_file=fitness_file)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
