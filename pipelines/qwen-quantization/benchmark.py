from pathlib import PurePath
import os
import sys
import json
from datetime import datetime
from benchmark_patches import patch_ifeval_evaluate, patch_mmlu_redux_result_metadata


def save_results_to_json(result: dict, test_option: str) -> str:
    result_dir = f"data/benchmarks/{result['model']}/{test_option}"
    os.makedirs(result_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(result_dir, f"{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    return filename

def build_engine(model_dir: str, model_config: dict):
    import evalution as eval

    engine_kwargs = {
        "backend": "auto",
        "device": "cuda:0",
    }

    if "exl3" in model_config.label.lower():
        print(f"[benchmark] EXL3 model detected: forcing fp16 to avoid exllamav3 dtype mismatch.")
        engine_kwargs["dtype"] = "float16"

    return eval.GPTQModel(**engine_kwargs)

def run_benchmark(model_dir: str, benchmarks: str) -> None:
    import evalution as eval

    patch_mmlu_redux_result_metadata()
    patch_ifeval_evaluate()

    # missing: LiveBench 0831, MATH, MultiPL-E
    TEST_OPTIONS = {
        "mmlu_pro": eval.benchmarks.mmlu_pro(apply_chat_template=True, batch_size=4),
        "gsm8k": eval.benchmarks.gsm8k(apply_chat_template=True, batch_size=4),
        "mmlu_redux": eval.benchmarks.mmlu_redux(batch_size=4),
        "GPQA": eval.benchmarks.GPQA(batch_size=4),
        "HumanEval": eval.benchmarks.HumanEval(batch_size=4),
        "MBPP": eval.benchmarks.MBPP(batch_size=4, dataset_path="google-research-datasets/mbpp"),
        "LiveCodeBench": eval.benchmarks.LiveCodeBench(batch_size=4),
        "IFEval": eval.benchmarks.IFEval(batch_size=4),
    }

    if not benchmarks:
        benchmarks = ",".join(TEST_OPTIONS.keys())

    model_cfg = eval.Model(path=model_dir, label=PurePath(model_dir).parts[-1])
    #engine = eval.GPTQModel(backend="auto", device="cuda:0", dtype="float16")
    engine = build_engine(model_dir, model_cfg)
    result_files = []

    for benchmark in benchmarks.split(","):
        test_option = benchmark.strip()

        test = TEST_OPTIONS.get(test_option)
        if test is None:
            print(f"Benchmark '{test_option}' is not recognized. Skipping.")
            continue

        run = eval.run(
            model=model_cfg,
            engine=engine,
            tests=[test],
        )

        result = {
            "model": model_cfg.label,
            "engine": run.engine,
            "result": run.tests[0].to_dict()
        }
        filename = save_results_to_json(result, test_option)
        result_files.append(filename)
    return result_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 benchmark.py <model_dir> <benchmarks>")
        print("benchmarks (optional - comma separated): mmlu_pro, gsm8k, mmlu_redux, GPQA, HumanEval, MBPP, LiveCodeBench, IFEval")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    benchmarks = sys.argv[2] if len(sys.argv) > 2 else None


    result_files = run_benchmark(model_dir, benchmarks)
    log = "Results saved to:"
    for filename in result_files:
        log += f"\n{filename}"
    print(log)