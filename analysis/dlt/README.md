# DLTlib cross-check

Independent validation of the `HybridEngine` static split against the **reference DLT
library** (DLTlib, from Barlas, *Multicore and GPU Programming, 2e*, Appendix G — the
same book §11.3 the engine is based on). It answers: *is our hand-rolled
`part_i = R_i / ΣR_j` actually the library's optimum?*

## Build & run

```bash
DLTLIB=/path/to/Multicore_and_GPU_2e_code/Chapter_11_Loadbalancing/DLTlib ./build.sh

# Manual mode: pass measured rates (Gcells/s, or any consistent unit)
./dlt_split iGPU=2.1 dGPU=31.5                  # comm-free optimum = R_i/ΣR_j
./dlt_split --e0 iGPU=10000 iGPU=2.1 dGPU=31.5  # add a fixed load-overhead to a node
./dlt_split CPU=158 GPU=27286                   # the CPU+GPU calibration case

# Fit mode: measure real per-step overhead, then predict the split at any grid size
./measure_overhead.sh                           # -> overhead.csv (device,cells,t_step_s)
./dlt_split --fit overhead.csv --target-N 16384 --target-N 32768
```

Needs GLPK (`-lglpk`) and the DLTlib sources. DLTlib's stock `random.c` has a bad
absolute include on line 21 — it must read `#include "random.h"` to compile.

## How the mapping works

DLTlib solves a single-level tree (a distributor root + compute children). We map:

| gol / HybridEngine            | DLTlib `InsertNode`                          |
| ----------------------------- | -------------------------------------------- |
| load-originating node (no compute) | `LON`, `power=1e12` (huge ⇒ ~0 load), a pure distributor |
| a compute node with rate `R_i` (cells/s) | child with `power = 1/R_i` |
| calibrated split `part_i`     | `Node::part` after `SolveImage(L, 1, c)`     |
| per-step launch/sync overhead | child `e0` (fixed cost)                      |

Only the *ratio* of powers matters, so units of `R` and `L` are free. With `link = 0`
(no communication) DLTlib's optimum is exactly `R_i / ΣR_j`.

## Result — DLTlib confirms the engine

Communication-free, DLTlib reproduces the engine's calibrated split **to the digit**:

| case (recorded)        | node | engine `R_i/ΣR_j` | DLTlib `part` |
| ---------------------- | ---- | ----------------- | ------------- |
| iGPU+dGPU (2.1, 31.5)  | iGPU | 0.0625            | **0.062500**  |
|                        | dGPU | 0.9375            | **0.937500**  |
| CPU+GPU (158, 27286)   | CPU  | 0.00576 (0.6 %)   | **0.005757**  |
|                        | GPU  | 0.99424           | **0.994243**  |

So `part_i = R_i/ΣR_j` **is** the book's §11.3 optimum — the engine's auto-calibration
is the reference result, not an approximation.

### Fitting `e0` from measured overhead (fit mode)

DLTlib's execution-time model is `time_i = power_i·(part_i·L + e0_i)`, so `e0` is in
**load units** and a physical per-step overhead `τ_i` (seconds) maps to `e0_i = τ_i·R_i`.
`measure_overhead.sh` runs each device **solo** through the hybrid harness across board
sizes (including **small** ones, where overhead is a resolvable fraction of the step)
and records the mean per-step wall `t_step`. `--fit` then decouples the two parameters:
`R_i` = peak rate from the **largest** board (compute-dominated, fully boosted); `τ_i`
= mean residual `t_step − cells/R_i` over the **small** boards. (Plain OLS of `t` on
`cells` is leverage-biased by the largest point and inflates `τ` by ~100×.)

Measured on this box (GTX 1660 Ti Max-Q + UHD 630): `R_iGPU=2.12`, `R_dGPU=31.5`
Gcells/s (naive share **6.30 %**), `τ_iGPU≈23 µs`, `τ_dGPU≈6 µs`. Predicted vs the
fraction-sweep empirical optimum:

| N | naive `R_i/ΣR` | DLTlib + fitted `e0` | empirical peak |
| --- | --- | --- | --- |
| 8192  | 6.30 % | 6.25 % | ~6 % |
| 16384 | 6.30 % | 6.29 % | ~5 % |
| 24576 | 6.30 % | 6.30 % | ~5 % |
| 32768 | 6.30 % | 6.30 % | ~6 % |
| 40960 | 6.30 % | 6.30 % | ~5 % |

**The verdict: the overhead is negligible.** The fixed per-step overhead is only tens
of µs — three to four orders of magnitude below the per-step compute at the grid sizes
where the hybrid runs (N≥8192, step ≥ ms). Its correction to the split is `≤0.05 pp`
and vanishes as `1/N²`, so DLT predicts essentially the **naive 6.3 %** at every size.
The empirical 5–6 % optimum is that same 6.3 % **within the fraction sweep's 1 % grid
resolution and GPU-boost noise** (the throughput-vs-fraction curve is flat-topped, so
best-of-5 picks a winning grid point that wanders ±1 step). No overhead term — fixed
launch *or* the one-row seam — is large enough to shift the optimum to 5 %; the earlier
"below 6.3 % because of overhead" reading over-interpreted sweep noise.

Two methodology notes: (1) `R_dGPU` only reaches 31.5 with a long boost warm-up on
large boards — a short sweep reads ~27, which alone moves the *naive* share to ~7 %, so
boost/thermal variance (±10–15 %) is the real first-order uncertainty. (2) A proportional
`--link` moves the share the *wrong* way (up toward 50/50): DLTlib's link is a
load-proportional transfer cost, which the fixed one-row GoL seam is not — overhead
belongs in `e0`, not `link`.
