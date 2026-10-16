#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${PACKAGE_ROOT}/outputs/logs"

jsub -J siuo-package-gen -q gpu_6000ada -n 8 -gpgpu 1 \
    -R 'span[hosts=1]' -cwd "${PACKAGE_ROOT}" \
    -o "${PACKAGE_ROOT}/outputs/logs/siuo-gen.%J.out" \
    -e "${PACKAGE_ROOT}/outputs/logs/siuo-gen.%J.err" \
    "${PACKAGE_ROOT}/scripts/run_siuo_generate_job.sh"
