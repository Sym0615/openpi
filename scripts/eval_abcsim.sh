#!/usr/bin/env bash
set -Eeuo pipefail

OPENPI_WORKDIR="${OPENPI_WORKDIR:-/sunyiming/openpi}"
cd "${OPENPI_WORKDIR}"
exec bash scripts/eval_abc_bottles_openpi.sh --config-name pi05_abc_sim "$@"
