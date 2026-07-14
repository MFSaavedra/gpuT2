#!/usr/bin/env bash
# sweep_fracs.sh -- iGPU+dGPU DLT fraction sweep.
#
# Holds the hybrid composition at igpu,dgpu and varies the iGPU row share, measuring
# steady-state kernel throughput at several grid sizes. This is the experiment behind
# report/img/hybrid_frac_sweep.png and the "From CPU+GPU to iGPU+dGPU" section of
# report/hybrid_report.tex: near the ~5-6% optimum a well-chosen static split beats
# pure dGPU by a margin that grows with N (the fixed one-row seam is amortized against
# N^2 compute), reaching ~99% of the R_iGPU+R_dGPU ceiling at N=32768.
#
# Explicit --fracs skips calibration, so each point is a clean static split; f=0 is
# pure dGPU (the iGPU is filtered out). Best-of-REPS per point (GPU boost variance is
# ~5-10%); G is scaled down at large N so wall-per-run stays bounded while the boost
# clocks still ramp. Output: one Pandas-ready CSV (the binary's 12-column --csv
# schema) with every raw row; the iGPU fraction is the `threads`-free extra we track
# via a leading column, so this script writes its own header.
#
# Env-overridable:  NS="8192 16384 32768"  FRACS="0 0.05 0.06"  REPS=5  ./scripts/sweep_fracs.sh
set -uo pipefail
GOL="${GOL:-./build/gol}"
OUT="${OUT:-results/linux/frac_sweep.csv}"
NS="${NS:-8192 16384 24576 32768 40960}"
FRACS="${FRACS:-0 0.03 0.04 0.05 0.06 0.07 0.08 0.10}"
REPS="${REPS:-5}"

# Per-N generation counts: fewer steps at large N (each step is dear, and the boost
# ramp is time-bounded so a handful of long steps already reach steady state).
gens_for() { local n="$1"; if   ((n<=8192));  then echo 300
                          elif ((n<=16384)); then echo 150
                          elif ((n<=24576)); then echo 100
                          elif ((n<=32768)); then echo 60
                          else                    echo 40; fi; }

if [[ ! -x "$GOL" ]]; then echo "error: '$GOL' not found (build with -DBUILD_CUDA=ON -DBUILD_OPENCL=ON)" >&2; exit 1; fi
mkdir -p "$(dirname "$OUT")"
echo "igpu_frac,$($GOL --csv-header)" > "$OUT"

# Warm up both GPU contexts before timing.
"$GOL" --engine cuda -r 4096 -c 4096 -g 50 >/dev/null 2>&1 || true
GOL_OCL_DEVICE=intel "$GOL" --engine opencl -r 1024 -c 1024 -g 20 >/dev/null 2>&1 || true

for n in $NS; do
  g="$(gens_for "$n")"
  for f in $FRACS; do
    df="$(awk "BEGIN{printf \"%.5f\", 1-$f}")"
    for ((r=1;r<=REPS;r++)); do
      row="$("$GOL" --engine hybrid --nodes igpu,dgpu --fracs "$f,$df" \
                    -r "$n" -c "$n" -g "$g" --csv 2>/dev/null | tail -1)"
      [[ -n "$row" ]] && echo "$f,$row" >> "$OUT"
    done
    # Peak mcells_kernel over the reps. Column 12 is mcells_kernel here: the leading
    # igpu_frac column shifts the binary's 12-field --csv schema right by one.
    best="$(awk -F, -v f="$f" '$1==f{if($12>m)m=$12} END{printf "%.1f", m}' "$OUT")"
    printf '[frac] N=%-6s igpu=%-5s -> best %s Mcells/s\n' "$n" "$f" "$best" >&2
  done
done
echo "wrote $(($(wc -l < "$OUT") - 1)) rows to $OUT" >&2
