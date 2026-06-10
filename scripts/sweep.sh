#!/usr/bin/env bash
#
# sweep.sh -- benchmark sweeps for the Game of Life CPU/CUDA backends.
#
# Produces two Pandas-ready CSVs under results/ (columns are the binary's own
# --csv schema: backend,rows,cols,generations,threads,block,shared,wrap,
# kernel_ms,wall_ms,mcells_kernel,mcells_wall):
#
#   results/sweep_cuda_opt.csv -- CUDA block-size x shared-memory sweep, large
#                                 grids. Feeds the report's optimization curves.
#   results/sweep_scaling.csv  -- CPU (varying threads) + CUDA (best config) vs
#                                 grid size N. Feeds CPU-vs-GPU, speed-up, and
#                                 the GPU-convenience-threshold analysis.
#
# Each config is run REPEATS times and every raw row is kept; the analysis takes
# the best (lowest-time) row per config -- matching "mejor tiempo" in the report.
# All runs use bounded edges, the deterministic random fill (--seed 1, identical
# work across backends), and the NullRenderer (no --renderer => transfers and
# drawing never enter the timed loop).
#
# A runtime guard predicts each config's wall time and reduces REPEATS (or skips)
# expensive runs, so the otherwise-quadratic CPU-sequential cost stays bounded.
#
# Everything is env-overridable, so the same script smoke-tests quickly, e.g.:
#   THREADS="1 2" BLOCKS="64 128" OPT_GRIDS="256" SCALE_GRIDS="128 256" \
#     REPEATS=2 ./scripts/sweep.sh
#
set -euo pipefail

# --- configuration (override via env) -------------------------------------
GOL="${GOL:-./build/gol}"
OUTDIR="${OUTDIR:-results}"
REPEATS="${REPEATS:-5}"          # max repeats per config (best-of in analysis)
SKIP_S="${SKIP_S:-45}"           # skip a config if one run is predicted slower than this

read -ra THREADS     <<< "${THREADS:-1 2 4 6 8 10 12}"          # CPU thread counts
read -ra BLOCKS      <<< "${BLOCKS:-32 64 128 256 512}"          # CUDA threads/block
read -ra OPT_GRIDS   <<< "${OPT_GRIDS:-1024 2048 4096 8192 16384}"
read -ra SCALE_GRIDS <<< "${SCALE_GRIDS:-64 128 256 512 1024 2048 4096 8192 16384}"
BEST_BLOCK="${BEST_BLOCK:-128}"  # CUDA config used for the scaling curve

# Generations per grid size N: many for small grids (so per-step overhead is
# visible above timer noise -- the point of the threshold analysis), few for
# large grids (so a run stays ~tens of ms..~1 s). Unlisted N fall back to a
# computed default in gens_for().
declare -A GENS=(
  [64]=2000 [128]=1000 [256]=500 [512]=200 [1024]=100
  [2048]=40 [4096]=20 [8192]=10 [16384]=5
)

# Rough throughput (cells/s) for the runtime guard only -- not reported.
RATE_CPU1=24000000        # ~24 Mcells/s on one core (flat, compute-bound)
RATE_CUDA=25000000000     # ~25 Gcells/s

CUDA_OPT_CSV="$OUTDIR/sweep_cuda_opt.csv"
SCALING_CSV="$OUTDIR/sweep_scaling.csv"

# --- helpers --------------------------------------------------------------

# gens_for N -> generation count for grid size N
gens_for() {
  local n="$1"
  if [[ -n "${GENS[$n]:-}" ]]; then
    echo "${GENS[$n]}"
  else
    awk -v n="$n" 'BEGIN{ g=int(2e8/(n*n)+0.5); if(g<5)g=5; if(g>2000)g=2000; print g }'
  fi
}

# cpu_rate THREADS -> estimated cells/s (cores scale linearly to 6 physical,
# then hyperthreading adds ~0.3x per logical core).
cpu_rate() {
  awk -v t="$1" -v r1="$RATE_CPU1" \
    'BEGIN{ eff = (t<=6) ? t : 6 + (t-6)*0.3; printf "%.0f", r1*eff }'
}

# plan_repeats PRED_SECONDS -> repeats to actually run (0 = skip this config)
plan_repeats() {
  awk -v p="$1" -v R="$REPEATS" -v skip="$SKIP_S" 'BEGIN{
    if (p > skip)     print 0;            # too slow: skip entirely
    else if (p > 10)  print 1;            # one run only
    else if (p > 2)   print (R>3?3:R);    # a few
    else              print R;            # full repeats
  }'
}

# run_repeats OUTFILE REP -- args after REP are passed to gol; appends REP rows.
run_repeats() {
  local out="$1" rep="$2"; shift 2
  local i
  for ((i = 1; i <= rep; i++)); do
    if ! "$GOL" "$@" --csv >> "$out"; then
      echo "  ! run failed, skipping rest of config: $* --csv" >&2
      return 1
    fi
  done
  return 0
}

# --- preflight ------------------------------------------------------------
if [[ ! -x "$GOL" ]]; then
  echo "error: '$GOL' not found or not executable." >&2
  echo "build it first: cmake -S . -B build -DBUILD_CUDA=ON && cmake --build build" >&2
  exit 1
fi
mkdir -p "$OUTDIR"

# Warm up the CUDA context once so the first timed run isn't paying init cost
# (best-of would also hide it, but this keeps even REPEATS=1 honest).
"$GOL" --engine cuda -r 256 -c 256 -g 5 >/dev/null 2>&1 || true

# --- experiment A: CUDA block x shared sweep ------------------------------
"$GOL" --csv-header > "$CUDA_OPT_CSV"
echo "== experiment A: CUDA block x shared (-> $CUDA_OPT_CSV) ==" >&2
for n in "${OPT_GRIDS[@]}"; do
  g="$(gens_for "$n")"
  cells=$(( n * n * g ))
  pred="$(awk -v c="$cells" -v r="$RATE_CUDA" 'BEGIN{ printf "%.3f", c/r }')"
  rep="$(plan_repeats "$pred")"
  [[ "$rep" -eq 0 ]] && { echo "[A] N=$n  SKIP (pred ${pred}s/run)" >&2; continue; }
  for block in "${BLOCKS[@]}"; do
    for shared in 0 1; do
      args=(--engine cuda --block "$block" -r "$n" -c "$n" -g "$g")
      [[ "$shared" -eq 1 ]] && args+=(--shared)
      printf '[A] N=%-5s block=%-3s shared=%s G=%-4s x%s\n' \
        "$n" "$block" "$shared" "$g" "$rep" >&2
      run_repeats "$CUDA_OPT_CSV" "$rep" "${args[@]}" || true
    done
  done
done

# --- experiment B: scaling (CPU threads + CUDA best) vs N ------------------
"$GOL" --csv-header > "$SCALING_CSV"
echo "== experiment B: scaling vs N (-> $SCALING_CSV) ==" >&2
for n in "${SCALE_GRIDS[@]}"; do
  g="$(gens_for "$n")"
  cells=$(( n * n * g ))

  # CPU, swept over thread counts
  for t in "${THREADS[@]}"; do
    rate="$(cpu_rate "$t")"
    pred="$(awk -v c="$cells" -v r="$rate" 'BEGIN{ printf "%.3f", c/r }')"
    rep="$(plan_repeats "$pred")"
    if [[ "$rep" -eq 0 ]]; then
      printf '[B] cpu  N=%-5s t=%-2s SKIP (pred %ss/run)\n' "$n" "$t" "$pred" >&2
      continue
    fi
    printf '[B] cpu  N=%-5s t=%-2s G=%-4s x%s (pred %ss/run)\n' \
      "$n" "$t" "$g" "$rep" "$pred" >&2
    run_repeats "$SCALING_CSV" "$rep" --engine cpu --threads "$t" \
      -r "$n" -c "$n" -g "$g" || true
  done

  # CUDA, best config
  pred="$(awk -v c="$cells" -v r="$RATE_CUDA" 'BEGIN{ printf "%.3f", c/r }')"
  rep="$(plan_repeats "$pred")"
  if [[ "$rep" -ne 0 ]]; then
    printf '[B] cuda N=%-5s block=%-3s G=%-4s x%s\n' "$n" "$BEST_BLOCK" "$g" "$rep" >&2
    run_repeats "$SCALING_CSV" "$rep" --engine cuda --block "$BEST_BLOCK" \
      -r "$n" -c "$n" -g "$g" || true
  fi
done

# --- summary --------------------------------------------------------------
echo "== done ==" >&2
printf 'rows written: %s (%s), %s (%s)\n' \
  "$(($(wc -l < "$CUDA_OPT_CSV") - 1))" "$CUDA_OPT_CSV" \
  "$(($(wc -l < "$SCALING_CSV") - 1))" "$SCALING_CSV" >&2
