#!/bin/bash

REPOSITORY=arthuravianna
LOCAL_DIR=/app/llm

GGUF_4BIT=Qwen2.5-14B-Instruct-GGUF-4bit
GGUF_8BIT=Qwen2.5-14B-Instruct-GGUF-8bit
GPTQ_4BIT=Qwen2.5-14B-Instruct-GPTQ-4bit
GPTQ_8BIT=Qwen2.5-14B-Instruct-GPTQ-8bit
AWQ_4BIT=Qwen2.5-14B-Instruct-AWQ-4bit
EXL3_4BIT=Qwen2.5-14B-Instruct-EXL3-4bit
EXL3_8BIT=Qwen2.5-14B-Instruct-EXL3-8bit

echo "Starting benchmark..."

hf download $REPOSITORY/$GGUF_4BIT --local-dir $LOCAL_DIR/$GGUF_4BIT
python3 benchmark.py $LOCAL_DIR/$GGUF_4BIT mmlu_redux
rm -r $LOCAL_DIR/$GGUF_4BIT

hf download $REPOSITORY/$GGUF_8BIT --local-dir $LOCAL_DIR/$GGUF_8BIT
python3 benchmark.py $LOCAL_DIR/$GGUF_8BIT mmlu_redux
rm -r $LOCAL_DIR/$GGUF_8BIT

hf download $REPOSITORY/$GPTQ_4BIT --local-dir $LOCAL_DIR/$GPTQ_4BIT
python3 benchmark.py $LOCAL_DIR/$GPTQ_4BIT mmlu_redux
rm -r $LOCAL_DIR/$GPTQ_4BIT

hf download $REPOSITORY/$GPTQ_8BIT --local-dir $LOCAL_DIR/$GPTQ_8BIT
python3 benchmark.py $LOCAL_DIR/$GPTQ_8BIT mmlu_redux
rm -r $LOCAL_DIR/$GPTQ_8BIT

hf download $REPOSITORY/$AWQ_4BIT --local-dir $LOCAL_DIR/$AWQ_4BIT
python3 benchmark.py $LOCAL_DIR/$AWQ_4BIT mmlu_redux
rm -r $LOCAL_DIR/$AWQ_4BIT

hf download $REPOSITORY/$EXL3_4BIT --local-dir $LOCAL_DIR/$EXL3_4BIT
python3 benchmark.py $LOCAL_DIR/$EXL3_4BIT mmlu_redux
rm -r $LOCAL_DIR/$EXL3_4BIT

hf download $REPOSITORY/$EXL3_8BIT --local-dir $LOCAL_DIR/$EXL3_8BIT
python3 benchmark.py $LOCAL_DIR/$EXL3_8BIT mmlu_redux
rm -r $LOCAL_DIR/$EXL3_8BIT

echo "Benchmark completed."