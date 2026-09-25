#!/usr/bin/env bash
set -euo pipefail
: "${PROFILE_OUTPUT:?PROFILE_OUTPUT must point to the .nsys-rep output prefix}"
exec nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --trace-fork-before-exec=true \
  --output "${PROFILE_OUTPUT}" \
  "$@"
