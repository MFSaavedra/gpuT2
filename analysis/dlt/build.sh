#!/usr/bin/env bash
# Build dlt_split against the book's DLTlib (Appendix G) + GLPK.
#
# DLTlib is a unity-include library: dlt_split.cpp #includes dltlib.cpp directly,
# so we only need its directory on the include path and -lglpk to link.
# Point DLTLIB at wherever the book's DLTlib sources live.
set -euo pipefail

DLTLIB="${DLTLIB:-$HOME/box/cpp/Multicore_and_GPU_2e_code/Chapter_11_Loadbalancing/DLTlib}"

if [[ ! -f "$DLTLIB/dltlib.cpp" ]]; then
  echo "error: DLTlib not found at '$DLTLIB'. Set DLTLIB=/path/to/DLTlib and re-run." >&2
  exit 1
fi

# Note: the stock random.c ships with a hard-coded absolute include on line 21
# (#include "/papers/cpp_lib/random.h"); it must read #include "random.h" instead
# for this to compile. (One-line fix in the DLTlib checkout.)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
g++ -O2 -I"$DLTLIB" "$here/dlt_split.cpp" -lstdc++ -lglpk -o "$here/dlt_split"
echo "built $here/dlt_split  (DLTlib=$DLTLIB)"
