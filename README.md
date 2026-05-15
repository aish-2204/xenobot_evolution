# Xenobot Soft Robot Evolution

Recreating the Kriegman et al. (2020) Xenobot pipeline using Python.
An evolutionary algorithm searches an 8×8×8 voxel space to find body plans
that maximise locomotion — no brain, no controller, just shape.

## Quickstart
```bash
git clone <your-repo>
cd xenobot_evolution
pip install -r requirements.txt
pytest tests/ -v
python -m src.milestone1.genome   # sanity check genome ops
```

## Milestones
| # | Focus | Est. Time |
|---|-------|-----------|
| M1 | Genome, fitness, serializer | 3–4 hrs |
| M2 | EA loop, operators, diagnostics | 4–5 hrs |
| M3 | VoxCraft-sim, ablation study | 4–5 hrs + compute |
| M4 | NSGA-II, Pareto, extension | 4–5 hrs |

## References
- Kriegman et al. (2020) — A scalable pipeline for designing reconfigurable organisms. PNAS.
- VoxCraft-sim: https://github.com/voxcraft/voxcraft-sim
