#!/bin/bash
# Minimal launcher for CEDI examiner.
#
# Usage: bash scripts/run_CEDI_examiner.sh <model_path> <output_dir> [<num_samples>]
#
# Required env: OPENAI_API_KEY (any OpenAI-compatible chat key)
# Optional env: OPENAI_BASE_URL (defaults to api.openai.com)
#               GEMINI_API_KEY / GEMINI_API_BASE (only if using gemini/<model> examinees)
#               HF_TOKEN (only for gated HuggingFace repos like google/gemma-3-12b-it)

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path> <output_dir> [<num_samples>]" >&2
    echo "Examples:" >&2
    echo "  $0 llava-hf/llava-1.5-7b-hf out/ 100" >&2
    echo "  $0 OpenGVLab/InternVL3-8B-Instruct out/ 100" >&2
    echo "  $0 uniapi/gpt-4o out/ 100         # API examinee" >&2
    echo "  $0 gemini/gemini-2.5-flash out/ 100" >&2
    exit 2
fi

MODEL_PATH="$1"
OUTPUT_DIR="$2"
NUM_SAMPLES="${3:-100}"
DATASET="${DATASET:-vg}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
NAME="$(basename "${MODEL_PATH}")"
OUTFILE="${OUTPUT_DIR}/${NAME}.json"
CACHE="${OUTFILE%.json}_cache.json"

# Ensure imports resolve when invoked from repo root
export PYTHONPATH="${PYTHONPATH:-}:./"

echo "[$(date +'%H:%M:%S')] Running CEDI examiner"
echo "  model: ${MODEL_PATH}"
echo "  dataset: ${DATASET}, num_samples: ${NUM_SAMPLES}"
echo "  outfile: ${OUTFILE}"

python examiner/CEDI_examiner.py \
    --dataset "${DATASET}" \
    --num_samples "${NUM_SAMPLES}" \
    --model_path "${MODEL_PATH}" \
    --outfile "${OUTFILE}" \
    --cache_file "${CACHE}"
