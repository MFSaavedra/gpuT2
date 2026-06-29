#!/usr/bin/env bash
# Launch gol_gui on the NVIDIA GPU to enable the ZERO-COPY CUDA/GL interop path.
#
# This box is an Optimus laptop: OpenGL otherwise defaults to the Intel iGPU, on
# which CUDA/GL interop is unavailable (cudaGraphicsGLRegisterBuffer -> "invalid
# OpenGL context"). Running bare still works -- the viewer falls back to host-upload
# display -- but the PRIME-offload env vars put the GL context on the NVIDIA GPU so
# the zero-copy interop path is used instead. QT_QPA_PLATFORM=xcb routes through
# Xwayland so the GLX offload vars apply.
#
# Usage: scripts/run_gui.sh [COLSxROWS] [--engine cpu|cuda] [--wrap] [--seed N] [--block N]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bin=""
for d in build build-gui; do
  if [[ -x "$root/$d/gol_gui" ]]; then bin="$root/$d/gol_gui"; break; fi
done

if [[ -z "$bin" ]]; then
  echo "error: gol_gui not found in build/ or build-gui/." >&2
  echo "Build it first:" >&2
  echo "  cmake -S . -B build -DBUILD_CUDA=ON -DBUILD_GUI=ON" >&2
  echo "  cmake --build build --target gol_gui -j" >&2
  exit 1
fi

exec env \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  QT_QPA_PLATFORM=xcb \
  "$bin" "$@"
