#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  scripts/quantize_hf_to_gguf_q8_0.sh \
    --llama-cpp-dir /workspace/llama.cpp \
    --hf-model-dir /workspace/models/Qwen2.5-7B-Instruct \
    --output-dir /workspace/models/Qwen2.5-7B-Instruct-GGUF-Q8_0

Converte um checkpoint Hugging Face original (BF16/FP16) para GGUF F16 e
depois usa llama-quantize para produzir Q8_0. Recusa checkpoints que já
declaram quantization_config, evitando uma dupla quantização silenciosa.
EOF
}

LLAMA_CPP_DIR=""
HF_MODEL_DIR=""
OUTPUT_DIR=""
THREADS="${LLAMA_QUANTIZE_THREADS:-$(nproc 2>/dev/null || printf '1')}"
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-cpp-dir) LLAMA_CPP_DIR="$2"; shift 2 ;;
    --hf-model-dir) HF_MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Argumento desconhecido: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$LLAMA_CPP_DIR" || -z "$HF_MODEL_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

CONVERTER="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZE="$LLAMA_CPP_DIR/build/bin/llama-quantize"
CONFIG="$HF_MODEL_DIR/config.json"

[[ -f "$CONVERTER" ]] || { printf 'Conversor ausente: %s\n' "$CONVERTER" >&2; exit 1; }
[[ -x "$QUANTIZE" ]] || { printf 'llama-quantize ausente ou sem permissão de execução: %s\n' "$QUANTIZE" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { printf 'config.json ausente: %s\n' "$CONFIG" >&2; exit 1; }

"$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
if config.get("quantization_config"):
    print("O checkpoint de entrada já declara quantization_config.", file=sys.stderr)
    print("Use o checkpoint Hugging Face original BF16/FP16; não faça dupla quantização.", file=sys.stderr)
    raise SystemExit(1)
PY

mkdir -p "$OUTPUT_DIR"
MODEL_NAME="$(basename "$HF_MODEL_DIR")"
F16="$OUTPUT_DIR/${MODEL_NAME}-F16.gguf"
Q8="$OUTPUT_DIR/${MODEL_NAME}-Q8_0.gguf"

printf 'Fonte HF: %s\n' "$HF_MODEL_DIR"
printf 'GGUF intermediário: %s\n' "$F16"
printf 'GGUF final: %s\n' "$Q8"
printf 'Threads de quantização: %s\n' "$THREADS"

"$PYTHON_BIN" "$CONVERTER" "$HF_MODEL_DIR" --outfile "$F16" --outtype f16
"$QUANTIZE" "$F16" "$Q8" Q8_0 "$THREADS"

printf '\nArquivo GGUF final:\n'
ls -lh "$Q8"
printf 'Assinatura: '
head -c 4 "$Q8" | od -An -tc
printf 'SHA-256:\n'
sha256sum "$Q8"
