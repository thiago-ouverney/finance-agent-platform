#! /bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_path> <bits>"
    exit 1
fi

MODEL_PATH="$1"
BITS="$2"
GPTQ_PATH="${MODEL_PATH}-GPTQ-${BITS}bit"
GGUF_PATH="${MODEL_PATH}-GGUF-${BITS}bit"
AWQ_PATH="${MODEL_PATH}-AWQ-${BITS}bit"
EXL3_PATH="${MODEL_PATH}-EXL3-${BITS}bit"

echo "Quantizing $MODEL_PATH"
if [ ! -d "$GPTQ_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$GPTQ_PATH" gptq "$BITS"
else
    echo "Skipping GPTQ: $GPTQ_PATH already exists"
fi

if [ ! -d "$GGUF_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$GGUF_PATH" gguf "$BITS"
else
    echo "Skipping GGUF: $GGUF_PATH already exists"
fi

if [ ! -d "$AWQ_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$AWQ_PATH" awq 4
else
    echo "Skipping AWQ: $AWQ_PATH already exists"
fi

if [ ! -d "$EXL3_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$EXL3_PATH" exl3 "$BITS"
else
    echo "Skipping EXL3: $EXL3_PATH already exists"
fi