# Qwen2.5 Quantization

> **Objective:** generate and evaluate local quantized Qwen2.5 artifacts using
> the experimental GPTQ, GGUF, AWQ, and EXL3 pipeline.

This repository contains quantization experiments on the [Qwen2.5](https://qwen.ai/blog?id=qwen2.5-llm) LLM. The following sections describe the steps necessary to download, quantize, and benchmark the quantized model.

## Reproducible golden-dataset workflow

The supported project workflow uses the root Make targets and a private CSV
copied to `/workspace/data/mopep-golden/train.csv`. It resolves a mutable model
selector such as `main` to an immutable Hugging Face commit, deterministically
selects calibration examples, renders the versioned 20-tag MOPEP prompt through
the Qwen chat template, saves provenance and weight hashes in
`quantization_manifest.json`, and keeps upload as a separate explicit step.

From a prepared RunPod checkout:

```shell
export QUANT_ENV_FILE=/workspace/quantization.env
make quantization-doctor
make quantization-setup
make quantization-data-check
make smoke-quantize-model
make run-quantize-model
make verify-quantized-model
make upload-quantized-model
```

Copy `pipelines/qwen-quantization/quantization.env.example` to
`/workspace/quantization.env` for the
non-secret experiment configuration. `HF_TOKEN` must be supplied only through
the environment (preferably a RunPod secret), never through that file or a Make
argument. The smoke uses Qwen2.5-0.5B, GPTQ 4-bit, and 8 golden train examples;
the default full run uses Qwen2.5-7B and 256 train examples. Python 3.11-3.13
and an NVIDIA R580+ RunPod image are required. The supported GPTQ path uses the
Torch backend; experimental AWQ/EXL3 additionally require a CUDA devel image
with `nvcc`. Publication is private by default,
refuses a non-empty repository unless `QUANT_HF_REPLACE_EXISTING=1`, and requires
both `QUANT_HF_PRIVATE=0` and `QUANT_HF_ALLOW_PUBLIC=1` for a public repo.

See [`docs/guias/gerar-modelo-quantizado.md`](../../docs/guias/gerar-modelo-quantizado.md)
for the empty-Pod bootstrap, private data transfer, token setup, verification,
and private Hub publication commands.

## Reproducible GGUF/iMatrix comparison

The llama.cpp workflow builds three single-file artifacts from the same pinned
Qwen2.5-7B BF16 GGUF source:

1. `Q4_K_M` without an importance matrix;
2. `Q4_K_M` with an importance matrix computed from 256 deterministic
   `train.csv` examples rendered through the Qwen chat template;
3. `Q8_0` as a higher-fidelity reference.

Comparing the two Q4 files isolates the effect of the iMatrix. Q8_0 is a
reference, not an iMatrix pair: llama.cpp does not apply importance weights to
the Q8_0 quantizer. The Hugging Face checkpoint is a build input, while all
three comparison artifacts and the BF16 intermediate are GGUF files. The
private rendered corpus is stored under `/workspace/data/quantization`, never
inside a model repository.

```shell
export QUANT_ENV_FILE=/workspace/quantization.env
make -C pipelines/qwen-quantization gguf-doctor
make -C pipelines/qwen-quantization gguf-setup
make -C pipelines/qwen-quantization gguf-smoke-build
make -C pipelines/qwen-quantization gguf-build-comparison
make -C pipelines/qwen-quantization gguf-verify-comparison
```

The build pins both the Hugging Face source revision and the llama.cpp commit,
records hashes for the dataset selection, corpus, iMatrix, BF16 intermediate,
and final files, refuses `test.csv`, and never requantizes an already quantized
checkpoint. A per-stage progress manifest prevents an interrupted build from
silently legitimizing a replaced intermediate on resume. Runtime comparison is
a separate, sequential 3×3 matrix in
`services/inference-runtime`; it records the exact GGUF SHA for every cell and
requires the build manifest, records the tokenizer metadata hash, and does not
fall back to another format when a runtime rejects a file.

See [`docs/guias/gerar-gguf-imatrix.md`](../../docs/guias/gerar-gguf-imatrix.md)
for the complete RunPod commands and the vLLM, llama.cpp, and Ollama smoke and
benchmark sequence.

This GGUF workflow deliberately stops after local build, verification, and
runtime comparison. It has no Hub upload target yet; do not reuse the legacy
single-Q8 upload command for this three-artifact set.

The sections below document the original experimental helper. Without
`--calibration-csv`, it deliberately retains the legacy MMLU calibration
fallback.

## Step 1 - Model download
First, download a Qwen2.5 version (e.g., 3B, 7B, 32B, etc). We recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli). The example below showcases how to download the Qwen2.5 3B-Instruct model from the Hugging Face repository using the CLI.
``` shell
hf download Qwen/Qwen2.5-7B-Instruct --revision <commit> --local-dir <path-to-model>
```

## Step 2 - Quantization
We defined four quantization strategies (all using 4-bit): GPTQ, GGUF, AWQ, and EXL3. To generate the quantized versions of the model, use the helper script below. Each quantized model is stored as `<path-to-model>-<STRATEGY>-4bit`, for example `Qwen2.5-7B-Instruct-GPTQ-4bit`.

``` shell
./quantize-qwen2.5.sh <path-to-model> 4
```

OR

```shell
docker run --rm --gpus all --env HF_TOKEN -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint /app/quantize-qwen2.5.sh viannaarthur/quantize-qwen2.5 /usr/llm/Qwen2.5-0.5B-Instruct 4
```

## Step 3 - Benchmark
Run the benchmark for a given model. To properly compare the quantized models with the base model, the benchmark script runs the same tests described in the [Qwen2 benchmark.5 technical report](https://arxiv.org/pdf/2412.15115). The benchmarks executed are listed below by category.
- General Tasks
    - MMLU-Pro
    - MMLU-redux
- Mathematics & Science Tasks
    - GPQA
    - GSM8K
- Coding Tasks
    - HumanEval
    - MBPP
    - LiveCodeBench
- Alignment Tasks
    - IFEval
    - Arena-Hard
    - MTbench

Some benchmarks, like GPQA, may used gated datasets. Therefore, set a Hugging Face access token to be able to run all tests.

``` shell
read -rsp 'Hugging Face token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

To execute the benchmark, run the Python script as below. The benchmark results are stored in the "benchmark" directory, which is created automatically in the same folder as the benchmark Python script.

``` shell
python3 benchmark.py <path-to-quantized-model>
```

OR

``` shell
docker run --rm --gpus all --env HF_TOKEN -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint python3 viannaarthur/quantize-qwen2.5 benchmark.py /usr/llm/Qwen2.5-0.5B-Instruct-GPTQ-4bit mmlu_redux
```

## Step 4 - Quality Evaluation
Measure perplexity in the quantization pipeline. TTFT, throughput and KV-cache behavior are measured by `services/inference-runtime`, where all runtimes use the same client protocol.

``` shell
python3 perplexity.py <path-to-quantized-model>
```

OR

``` shell
docker run --rm --gpus all --env HF_TOKEN -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint python3 viannaarthur/quantize-qwen2.5 perplexity.py /usr/llm/Qwen2.5-0.5B-Instruct-GPTQ-4bit
```
