#!/usr/bin/env bash
# measure_overhead.sh -- per-device affine timing for the DLT e0 fit.
#
# Runs each compute node SOLO through the hybrid harness (--nodes X --fracs 1) at
# several board sizes and records the mean per-step wall time (kernel_ms/gens; for
# the hybrid engine kernel_ms is the accumulated per-step wall = launch+sync). Solo
# runs have no seam, so the fitted intercept is the device's fixed per-step overhead
# (launch/sync); the slope is 1/R. Best (min) of REPS per point to pin the boosted
# clock, matching the report's best-of-5 methodology.
#
# Output CSV (device,cells,t_step_s) -> analysis/dlt/overhead.csv, consumed by
#   ./dlt_split --fit overhead.csv --target-N <N>
set -uo pipefail
GOL="${GOL:-./build/gol}"
OUT="${OUT:-analysis/dlt/overhead.csv}"
NODES="${NODES:-igpu dgpu}"          # solo device tokens to profile
NS="${NS:-1024 2048 4096 8192 16384}"
REPS="${REPS:-3}"

gens_for() { local n="$1"; if   ((n<=2048)); then echo 100
                          elif ((n<=8192)); then echo 40
                          else                   echo 20; fi; }

[[ -x "$GOL" ]] || { echo "error: '$GOL' not found (build -DBUILD_CUDA=ON -DBUILD_OPENCL=ON)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
echo "device,cells,t_step_s" > "$OUT"

# Warm up both contexts (boost ramp + JIT).
"$GOL" --engine hybrid --nodes dgpu --fracs 1 -r 4096 -c 4096 -g 30 >/dev/null 2>&1 || true
"$GOL" --engine hybrid --nodes igpu --fracs 1 -r 1024 -c 1024 -g 20 >/dev/null 2>&1 || true

for node in $NODES; do
  for n in $NS; do
    g="$(gens_for "$n")"
    best_ms=""
    for ((r=1;r<=REPS;r++)); do
      km="$("$GOL" --engine hybrid --nodes "$node" --fracs 1 -r "$n" -c "$n" -g "$g" --csv 2>/dev/null \
              | tail -1 | awk -F, '{print $9}')"       # col 9 = kernel_ms (accumulated per-step wall)
      [[ -z "$km" ]] && continue
      # min kernel_ms over reps
      if [[ -z "$best_ms" ]] || awk "BEGIN{exit !($km<$best_ms)}"; then best_ms="$km"; fi
    done
    [[ -z "$best_ms" ]] && { echo "warn: no data for $node N=$n" >&2; continue; }
    t_step_s="$(awk "BEGIN{printf \"%.9g\", ($best_ms/1000.0)/$g}")"
    cells="$((n*n))"
    echo "$node,$cells,$t_step_s" >> "$OUT"
    printf '[measure] %-5s N=%-6s cells=%-10s t_step=%s s\n' "$node" "$n" "$cells" "$t_step_s" >&2
  done
done
echo "wrote $OUT" >&2
