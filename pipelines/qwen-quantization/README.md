# Qwen2.5 Quantization
This repository contains quantization experiments on the [Qwen2.5](https://qwen.ai/blog?id=qwen2.5-llm) LLM. The following sections describe the steps necessary to download, quantize, and benchmark the quantized model.

## Step 1 - Model download
First, download a Qwen2.5 version (e.g., 3B, 7B, 32B, etc). We recommend using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli). The example below showcases how to download the Qwen2.5 3B-Instruct model from the Hugging Face repository using the CLI.
``` shell
hf download Qwen/Qwen2.5-3B-Instruct --local-dir <path-to-model>
```

## Step 2 - Quantization
We defined four quantization strategies (all using 4-bit): GPTQ, GGUF, AWQ, and EXL3. To generate the quantized versions of the model, use the helper script below. Each quantized model will be stored in a directory named `"<path-to-model>-4bit-<strategy-name>"`.

``` shell
./quantize-qwen2.5.sh <path-to-model> 4
```

OR

```shell
docker run --rm --gpus all -e HF_TOKEN=<hf_token> -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint /app/quantize-qwen2.5.sh viannaarthur/quantize-qwen2.5 /usr/llm/Qwen2.5-0.5B-Instruct 4
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
export HF_TOKEN <your Hugging Face token>
```

To execute the benchmark, run the Python script as below. The benchmark results are stored in the "benchmark" directory, which is created automatically in the same folder as the benchmark Python script.

``` shell
python3 benchmark.py <path-to-quantized-model>
```

OR

``` shell
docker run --rm --gpus all -e HF_TOKEN=<hf_token> -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint python3 viannaarthur/quantize-qwen2.5 benchmark.py /usr/llm/Qwen2.5-0.5B-Instruct-GPTQ-4bit mmlu_redux
```

## Step 4 - Performance Evaluation
Run a performance evaluation that consists of token metrics, such as time to first token (TTFT) and token per second (TPS), and perplexity. Both performance metrics are computed by the performance Python script and is executed as below:

``` shell
python3 performance.py <path-to-quantized-model>
```

OR

``` shell
docker run --rm --gpus all -e HF_TOKEN=<hf_token> -v <path-to-model>:/usr/llm -v <host-path-to-logs>:/app/data --entrypoint python3 viannaarthur/quantize-qwen2.5 performance.py /usr/llm/Qwen2.5-0.5B-Instruct-GPTQ-4bit
```