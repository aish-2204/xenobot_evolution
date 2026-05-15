# Xenobot Evolution — Project Analysis

Recreating the Kriegman et al. (2020) Xenobot paper: evolving soft robot body plans
using an evolutionary algorithm and physics simulation to maximise locomotion.

---

## What I Built (Plain English)

The project is divided into four milestones, each adding a layer on top of the last.

### Milestone 1 — The Building Blocks

**What I did:**
I created the foundation: a way to represent a robot, simulate it, and score it.

- **Genome** — A robot is stored as an 8×8×8 grid (512 tiny cubes called voxels). Each cube is one of four types: empty, structural/passive, active-expand (pushes outward), or active-contract (pulls inward). The combination of expanding and contracting cubes is what makes the robot move.
- **Serializer** — Converts the NumPy grid into a special XML format (VXA) that the physics simulator (voxelyze) can read. A lot of careful work went here because voxelyze is strict about its format — wrong tags or wrong material definitions cause silent failures.
- **Fitness evaluator** — Runs the voxelyze simulator as a subprocess, waits for it to finish, then reads the result file to get the robot's score (normalised centre-of-mass displacement — how far it travelled).
- **Visualiser** — Renders 3D images of robots using PyVista, and generates statistics about body shape using Matplotlib.

**Key insight I found:**
- The search space is 4^512 ≈ 10^308 possible robots. That's 10^228 times more possibilities than atoms in the observable universe. Brute-force search is physically impossible — evolution is not a nice-to-have, it is the only viable strategy.
- At 75% voxel density, almost every random robot is one connected body (99.8% of voxels stay after connectivity repair). So connectivity is rarely the problem.
- Simulation cost hits 1 second per robot at grid size N=3 (27 voxels). At N=8 (the full grid), each robot takes ~22 seconds. This means parallelism is mandatory for any serious run.

---

### Milestone 2 — The Evolutionary Engine

**What I did:**
I wired up a genetic algorithm using the DEAP framework with five genetic operators and ran experiments comparing them.

- **3 mutation operators:**
  - `random_flip` — randomly re-rolls individual voxels (many small independent changes)
  - `block` — replaces a small cube-shaped region (fewer but larger changes)
  - `grow_shrink` — expands or shrinks the outer surface of the body (changes shape, not material type)
- **2 crossover operators:**
  - `uniform_voxel` — each voxel is swapped between two parents with 50% probability
  - `one_point_slice` — everything above a random Z-plane is swapped between parents
- **Baseline run:** 20 robots, 30 generations, ~38 minutes. Best fitness reached: **24.81**

**Results I got:**

| Metric | What happened |
|--------|--------------|
| Fitness (gen 0) | ~2.7 average, 7.0 best |
| Fitness (gen 30) | ~23.8 average, 24.8 best |
| Diversity (gen 0) | 0.75 (very diverse) |
| Diversity (gen 30) | 0.016 (nearly all identical) |

Diversity collapsed fast — by generation 10 the population was already mostly clones of one good robot. This is premature convergence.

**Operator comparison (30 gens, 3 seeds each):**

| Operator | Best fitness range |
|----------|-------------------|
| `block` | 17.4 – 23.1 |
| `random_flip` | 15.9 – 18.7 |
| `grow_shrink` | 9.9 – 19.4 |

`block` was the most consistent winner. `grow_shrink` was the most variable — sometimes competitive, sometimes stuck.

**Tournament size experiment:**
- k=2 (weak selection): slow progress, diversity stays high, never converges well
- k=3 (default): good balance — decent speed, diversity lasts longer
- k=7 (strong selection): fast early gains, but diversity collapses immediately, stagnates

---

### Milestone 3 — Scaling Up with Parallelism

**What I did:**
Rewrote the evaluation pipeline to use multiple CPU cores simultaneously (multiprocessing). Ran a proper large-scale evolution. Then ran an ablation study — systematically removing one component at a time to see how much each one contributes.

**Full run results (pop=30, 100 gens, 4 workers):**

| Generation | Best fitness | Diversity |
|-----------|-------------|----------|
| 0 | 13.60 | 0.751 |
| 10 | 28.94 | 0.013 |
| 30 | 39.34 | 0.003 |
| 55 | 42.35 | 0.007 |
| 85 | 45.24 | 0.001 |
| 100 | **45.24** | 0.000 |

Second run (pop=20, 50 gens) reached **49.88** — showing smaller populations can still improve if you run more generations.

**Ablation study (50 gens, 2 seeds each):**

| Condition | What was removed | Best fitness | Conclusion |
|-----------|-----------------|-------------|-----------|
| baseline | nothing | ~22.7 | reference |
| no_crossover | crossover disabled | ~21.2 | crossover helps a little |
| no_connectivity | LCC repair skipped | ~22.3 | similar to baseline |
| no_mutation | mutation disabled | ~14.8 | mutation is essential for escape |
| random_selection | tournament replaced by random | ~5–8 | selection pressure is critical |
| single_material | only passive voxels allowed | **0.0** | active material is absolutely required |

**Biggest finding:** Without selection (random_selection), the algorithm barely moves. Without active material (single_material), the robot cannot move at all — fitness stays exactly 0 across all 50 generations. Without mutation (no_mutation), the algorithm converges immediately and never escapes the initial population quality.

---

### Milestone 4 — Multi-Objective Evolution + MAP-Elites

**What I did:**
Instead of maximising one thing (distance), I optimised two things simultaneously: maximise distance AND minimise the number of voxels used. This finds a range of robots — some move very far, some are very compact, and some are a good tradeoff. I used NSGA-II, the standard multi-objective evolutionary algorithm.

I also ran MAP-Elites, which is a "quality-diversity" method that tries to fill a grid of different robot behaviours (by distance and body symmetry) rather than converging on one best solution.

**NSGA-II results (pop=40, 80 gens):**

| Generation | Max distance | Pareto front size | Hypervolume |
|-----------|-------------|------------------|------------|
| 0 | 13.60 | 2 | 1,963 |
| 20 | 56.11 | 15 | 11,206 |
| 50 | 75.91 | 5 | 18,405 |
| 70 | 97.60 | 40 | 25,545 |
| 80 | **107.35** | 31 | **28,820** |

The best single-objective fitness jumped to **107 from M3's 49** — multi-objective search explored a much wider part of the fitness landscape before converging, and that exploration paid off.

Three types of robot were found on the Pareto front:
- **Most mobile** — uses many voxels, moves farthest
- **Most efficient** — uses very few voxels, still moves reasonably
- **Knee point** — the best balance between the two extremes

---

## What the Results Tell Us (Insights)

### 1. Evolution works, but premature convergence is the main enemy
In every run, diversity collapsed within 10–15 generations. After that, mutation was the only source of new variation. The algorithm continued to improve (it jumped from 39.3 to 42.4 at gen 55 after being stuck since gen 30), but only rarely and slowly. The core problem is not finding good solutions — it is staying diverse enough to keep exploring.

### 2. Active material arrangement matters more than body shape
The ablation confirms active material is non-negotiable (zero fitness without it). The fact that no_connectivity performed similarly to baseline suggests that at 75% fill density, disconnected voxels are rare anyway — the LCC constraint is a safety net but not a major factor.

### 3. Block mutation is the best single operator
It outperformed random_flip (which changes too many things at once) and grow_shrink (which changes shape but not material layout). When a good body is found, block mutation can refine a region without destroying the rest — a good exploitation strategy.

### 4. Multi-objective search is better than single-objective here
NSGA-II reached fitness 107 vs 49 for the M3 baseline. By simultaneously searching for efficient robots, it discovered body plans that single-objective search would have ignored (because they used fewer voxels). Some of those turned out to be excellent movers.

### 5. Selection pressure is critical but must be balanced
Random selection (no tournament) produces near-random search — fitness peaks around 8. Strong tournament (k=7) converges too fast and gets stuck. k=3 is the sweet spot for populations of 20–40.

---

## What Should Be Improved

### Bugs to fix

| Issue | Where | Fix |
|-------|-------|-----|
| `np.trapz` was removed in NumPy 2.x | `src/milestone3/ablation.py:283` | Replace with `np.trapezoid` (NumPy 2.0+) — this caused the ablation stats CSV to be empty |
| Ablation stats DataFrame is empty | `results/m3/ablation/ablation_stats.csv` | Re-run after fixing the trapz bug |
| Morphological computation section is blank | `notebooks/milestone4_notebook.ipynb` cells 16-17 | Fill in after running the morphological analysis |

### Algorithm improvements

**1. Add diversity maintenance**
The biggest weakness. Diversity collapses by gen 10–15 in every run. Options:
- **Island model** — split the population into 3–4 sub-populations, only occasionally exchange individuals. Each island finds different solutions.
- **Fitness sharing / niching** — penalise robots that are too similar to other high-fitness robots.
- **Novelty search** — reward robots for being different from anything seen before, not just for fitness.

**2. Adaptive mutation rate**
Currently `mutpb=0.3` is fixed. A simple improvement: increase mutation rate when diversity drops below a threshold, decrease it when diversity is high. This is called adaptive operator control.

**3. Grow-shrink operator tuning**
`grow_shrink` had the most variable results. The problem is that morphological erosion/dilation sometimes changes 0 voxels (if the body is already compact or fully hollow). Adding a fallback — if no change occurred, apply a small random flip — would make it more reliable.

**4. Warm-start from M3 best individuals**
The M4 NSGA-II run started from a random population. Seeding 20–30% of the initial population with the best genomes from M3 would give a much stronger starting point.

### Config improvements

```yaml
# nsga2.yaml — suggested changes
tournsize: 3          # currently 2 — raise to 3 for slightly more selection pressure
n_workers: 6          # if machine has 8+ cores, more parallelism helps
sim_time: 0.75        # currently mixed (0.5 in notebooks, 1.0 in final) — standardise
```

**Inconsistency to fix:** M2 notebooks use `sim_time=0.5`, M3 uses `sim_time=0.5` during ablation and `sim_time=1.0` for final renders. Results are not directly comparable across milestones. Pick one value and stick to it.

### Testing improvements

| Gap | Suggested test |
|-----|---------------|
| No test for diversity collapse detection | Add test: run 5 gens with k=7, assert diversity < 0.1 by gen 3 |
| No test for ablation CSV output | Add test: mock a short ablation run, assert stats CSV is non-empty |
| No test for NSGA-II Pareto size | Add test: after 10 gens, assert Pareto front has ≥ 2 individuals |
| `np.trapz` bug not caught | Add a test that imports `run_all_ablations` and calls it with 1 gen — would have caught the NumPy 2.x incompatibility immediately |

### What to run next (if time allows)

1. **Fix the np.trapz bug and re-run the ablation** — the current ablation stats are incomplete. This is the highest priority since it affects the report.
2. **Run NSGA-II for 200 gens** (currently only 80) to see if the Pareto front stabilises or keeps growing.
3. **Complete the MAP-Elites run** — the notebook only shows setup; no grid was filled during the run shown in the notebook. A full 500-iteration run would show whether diversity in behaviour space correlates with fitness.
4. **Fill in the morphological computation section** in milestone4_notebook — the relationship between active ratio, CoM height, symmetry, and fitness is currently a placeholder.

---

## Quick Reference: Key Numbers

| Milestone | Best fitness | Population | Generations | Wall time |
|-----------|-------------|-----------|------------|----------|
| M1 (manual checkerboard) | 4.37 | — | — | ~5s |
| M2 (baseline EA) | 24.81 | 20 | 30 | 38 min |
| M3 (parallel EA) | 49.88 | 20 | 50 | ~2–4 hrs |
| M4 (NSGA-II) | 107.35 | 40 | 80 | ~1–2 hrs |

The jump from M2→M3 came from more generations and parallelism.
The jump from M3→M4 came from multi-objective search opening up the landscape.
