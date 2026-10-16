#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-qwen25vl7b-grpo-step529}"
MODEL_DIR="${MODEL_DIR:-${PACKAGE_ROOT}/model/huggingface_step_529}"
DATASET="${PACKAGE_ROOT}/data/siuo_gen.json"
IMAGE_ROOT="${PACKAGE_ROOT}/data/images"
GEN_OUTPUT="${PACKAGE_ROOT}/outputs/test_results/siuo-gen-${MODEL_NAME}.json"

[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Model not found: ${MODEL_DIR}" >&2; exit 1; }
[[ -f "${DATASET}" ]] || { echo "Dataset not found: ${DATASET}" >&2; exit 1; }
[[ -d "${IMAGE_ROOT}" ]] || { echo "Images not found: ${IMAGE_ROOT}" >&2; exit 1; }

RANGE_ARGS=(--start "${SIUO_START:-0}")
if [[ -n "${SIUO_END:-}" ]]; then
    RANGE_ARGS+=(--end "${SIUO_END}")
fi

"${PYTHON_BIN}" -u "${PACKAGE_ROOT}/eval/generate_siuo.py" \
    --model "${MODEL_DIR}" \
    --dataset "${DATASET}" \
    --image-root "${IMAGE_ROOT}" \
    --output "${GEN_OUTPUT}" \
    --num-responses 3 \
    --max-new-tokens 500 \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-pixels 401408 \
    --seed 42 \
    "${RANGE_ARGS[@]}"

echo "Generation complete: ${GEN_OUTPUT}"
