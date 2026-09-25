#!/usr/bin/env bash
set -euo pipefail
: "${PROFILE_OUTPUT:?PROFILE_OUTPUT must point to the .ncu-rep output prefix}"
exec ncu \
  --target-processes all \
  --set roofline \
  --launch-skip 0 \
  --launch-count 1 \
  --export "${PROFILE_OUTPUT}" \
  "$@"
