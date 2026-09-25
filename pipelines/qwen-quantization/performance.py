import datetime
import json
from pathlib import PurePath
import sys
import os
import time
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from gptqmodel import GPTQModel

DEFAULT_PERPLEXITY_WINDOW_SIZE = 4096
DEFAULT_PERPLEXITY_STRIDE = 512
DEFAULT_MAX_OUTPUT_TOKENS = 256


def load_model(model_dir):
    config = AutoConfig.from_pretrained(model_dir)
    quantization_config = getattr(config, "quantization_config", None) or {}
    quant_method = quantization_config.get("method", quantization_config.get("quant_method", ""))
    quant_method = str(quant_method).lower()

    if quant_method == "exl3":
        return GPTQModel.from_quantized(model_dir, device="cuda:0", dtype=torch.float16)

    if quant_method:
        return GPTQModel.from_quantized(model_dir, device="cuda:0")

    return AutoModelForCausalLM.from_pretrained(model_dir).to("cuda")

# Evaluate perplexity with a sliding context window to fit long corpora on GPU.
def compute_perplexity(model, tokenizer, window_size, stride):
    # 1. Load evaluation dataset (e.g., WikiText)
    test_data = load_dataset("closji/wikitext__wikitext-2-raw-v1", split="test")
    separator_ids = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    token_ids = []

    # Tokenize incrementally so the tokenizer never has to process the whole
    # evaluation corpus as one oversized sequence.
    for index, text in enumerate(test_data["text"]):
        if index > 0:
            token_ids.extend(separator_ids)

        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])

    encodings = {"input_ids": torch.tensor([token_ids], dtype=torch.long)}

    # 2. Define sliding window parameters
    model_max_length = model.config.max_position_embeddings
    if window_size is None:
        window_size = min(model_max_length, DEFAULT_PERPLEXITY_WINDOW_SIZE)

    seq_len = encodings["input_ids"].size(1)

    nlls = []
    prev_end_loc = 0

    # 3. Calculate negative log-likelihood (NLL)
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + window_size, seq_len)
        # Only score tokens that were not counted in the previous window.
        # The overlap is kept only to provide left-context to the model.
        trg_len = end_loc - prev_end_loc  # how many tokens we want to predict
        
        input_ids = encodings["input_ids"][:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context tokens in loss calculation

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 4. Compute Perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return {
        "setup": {
            "model_supported_window_size": model_max_length,
            "window_size": window_size,
            "stride": stride,
            "seq_len": seq_len
        },
        "result": ppl.item()
    }

# Measure a rough first-token proxy and sustained generation throughput.
def compute_token_generation_metrics(model, tokenizer, new_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    prompt = "Write a long essay on the Roman Empire."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_token_count = inputs["input_ids"].shape[1]

    # This is a proxy for TTFT because it times a full generate() call that
    # produces one token, rather than stopping exactly when the first token is emitted.
    # 1. Calculate time to first token
    start_time = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=1)
    end_time = time.perf_counter()
    time_to_first_token = end_time - start_time

    # 2. Generate 256 new tokens
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=new_tokens)
    end_time = time.perf_counter()

    # 3. Compute tokens per second
    total_output_tokens = outputs.shape[1] - input_token_count
    elapsed_time = end_time - start_time
    tokens_per_second = total_output_tokens / elapsed_time

    return {
        "time_to_first_token": time_to_first_token,
        "tokens_per_second": tokens_per_second,
        "total_output_tokens": total_output_tokens,
        "elapsed_time": elapsed_time
    }

if __name__ == "__main__":
    model_dir = None
    perplexity_window_size = DEFAULT_PERPLEXITY_WINDOW_SIZE
    perplexity_stride = DEFAULT_PERPLEXITY_STRIDE
    if len(sys.argv) == 2:
        model_dir = sys.argv[1]
    elif len(sys.argv) == 4:
        model_dir = sys.argv[1]
        perplexity_window_size = int(sys.argv[2])
        perplexity_stride = int(sys.argv[3])
    else:
        print("Usage: python3 performance.py <model_dir> <perplexity_window_size> <perplexity_stride>")
        print(f"perplexity_window_size (optional): the size of the sliding window used to compute perplexity, default is {DEFAULT_PERPLEXITY_WINDOW_SIZE}")
        print(f"perplexity_stride (optional): the stride of the sliding window used to compute perplexity, default is {DEFAULT_PERPLEXITY_STRIDE}")
        sys.exit(1)

    model_name = PurePath(model_dir).parts[-1]
    model = load_model(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    perplexity = compute_perplexity(model, tokenizer, window_size=perplexity_window_size, stride=perplexity_stride)
    tps_metrics = compute_token_generation_metrics(model, tokenizer)

    # save results to file
    result_dir = f"data/performance/{model_name}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(result_dir, f"{timestamp}.json")
    with open(filename, "w") as f:
        json.dump({
            "perplexity": perplexity,
            "token_generation_metrics": tps_metrics
        }, f, indent=4)