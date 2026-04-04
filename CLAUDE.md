# HRT/Partcl Macro Placement Challenge

## Goal

Win the $20,000 prize by developing a macro placement algorithm that beats SA and RePlAce baselines on proxy cost and OpenROAD flow metrics (WNS, TNS, Area).

**Deadline**: May 21, 2026, 11:59 PM Pacific

## Evaluation

### Proxy Cost (Tier 1 — all submissions)
```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```
- Evaluated on 17 IBM ICCAD04 benchmarks (ibm01–ibm18, no ibm05)
- Lower is better; zero overlaps required

### OpenROAD Flow (Tier 2 — top 7 by proxy)
- Evaluated on NG45 designs: ariane133, ariane136, mempool_tile, nvdla + 1-2 hidden
- Must beat both SA and RePlAce on WNS, TNS, and Area

### Baselines to beat
| Method | Avg Proxy Cost |
|--------|---------------|
| RePlAce | 1.4578 |
| SA | 2.1251 |

### Current leaderboard leaders
- UT Austin (DREAMPlace): **1.4076**
- BakaBobo (Spread+Refine): **1.4403**
- RePlAce baseline: **1.4578**

## Constraints

- Runtime: < 1 hour per benchmark (ideally < 5 min)
- Hardware: AMD EPYC 9655P, 16 cores, 100GB RAM, NVIDIA RTX 6000 Ada 48GB
- No overlaps in final placement
- No hardcoded solutions; algorithm must generalize
- Must use TILOS MacroPlacement evaluator as-is (no modification)
- Winning code must be open-sourced under Apache 2.0 or GPL

## Project Structure

```
macro_place/         # Core Python package
  benchmark.py       # Benchmark dataclass (PyTorch tensors)
  loader.py          # Load benchmarks from ICCAD04 format
  objective.py       # Proxy cost computation
  utils.py           # Validation and visualization
  def_writer.py      # DEF file export
submissions/
  examples/
    greedy_row_placer.py
    simple_random_placer.py
external/MacroPlacement/   # TILOS evaluator + ICCAD04 testcases (git submodule)
benchmarks/processed/      # Pre-processed .pt benchmark files
```

## Common Commands

```bash
# Run on single benchmark
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01

# Run on all IBM benchmarks
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Run on NG45 commercial designs
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize result
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

## Key API

```python
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

benchmark, plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/ibm01')
# benchmark.macro_positions: Tensor [N, 2] — (x, y) centers
# benchmark.macro_sizes: Tensor [N, 2] — (width, height)
# benchmark.macro_fixed: Tensor [N] — boolean fixed mask

costs = compute_proxy_cost(placement, benchmark, plc)
# Returns: proxy_cost, wirelength_cost, density_cost, congestion_cost, overlap_count
```

Placement submission must be a `[num_macros, 2]` tensor of (x, y) center positions.

## IBM Benchmark Scale
- 246–537 hard macros per benchmark
- 7,269–16,253 nets
- 43–53% area utilization
- Canvas ~22–34 μm²

## Our Submissions & Progress

### V1: Gravity Placer (`submissions/v1_gravity_placer.py`)
- Algorithm: FD attraction via adjacency matrix (80 iters) + spiral legalization
- Avg Proxy: **1.7153** (beats SA by 19%, loses to RePlAce by 18%)
- Strengths: Excellent wirelength (0.061 avg)
- Weaknesses: High congestion (2.18 avg = 63% of cost)

### V2: Gradient Placer (`submissions/v2_gradient_placer.py`)
- Algorithm: PyTorch Adam optimizer with multi-objective differentiable loss
  - HPWL via log-sum-exp on real net connectivity
  - Grid-based density penalty (matches TILOS top-10%)
  - TILOS congestion heatmap feedback every 100 steps (F.grid_sample)
  - Overlap penalty (pairwise hard-macro)
  - Dynamic weights: λ_den decays 2.0→0.3, λ_cong ramps 0.2→1.0
  - 500 Adam steps, spiral legalization
- Avg Proxy: **1.4522** — beats RePlAce baseline (1.4578) by 0.4%
- Beats SA on all 17 benchmarks (avg +31.7%)
- Beats or matches RePlAce on 9 of 17 benchmarks
- Runtime: ~92s/benchmark avg (within 5-min target)

### Leaderboard position (V2)
| Rank | Team | Avg Proxy |
|------|------|-----------|
| 1 | UT Austin (DREAMPlace) | 1.4076 |
| 2 | BakaBobo (Spread+Refine) | 1.4403 |
| **3** | **Us (V2 Gradient)** | **1.4522** |
| — | RePlAce baseline | 1.4578 |

### Tuning
- `scripts/meta_tuner.py` — Optuna hyperparameter search over λ schedules
- Env vars: PLACER_LR, PLACER_NUM_STEPS, PLACER_LAMBDA_DEN_START/END, PLACER_LAMBDA_CONG_START/END

### Key Technical Insights
- Congestion is 50-60% of proxy cost; density is 30%; wirelength only 5%
- Soft macro co-optimization HURTS density (learned from V2 prototyping)
- Global spreading doesn't help — congestion is structural (macro routing blockage)
- Gradient-based placement with real cost feedback outperforms pure heuristics
- The differentiable density grid (vectorized [N,R,C] overlap) is fast and accurate
