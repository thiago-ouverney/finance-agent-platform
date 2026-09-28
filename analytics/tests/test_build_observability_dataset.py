import hashlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from analytics.src.build_observability_dataset import (
    TABLE_NAMES,
    build_observability_dataset,
    discover_runs,
)


MODEL_SHA = "a" * 64
TOKENIZER_SHA = "b" * 64


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class BuildObservabilityDatasetTests(unittest.TestCase):
    def write_run(
        self,
        root: Path,
        name: str,
        *,
        status: str,
        experiment_id: str,
        runtime: str,
        model: str = "model-a",
        context: int = 8192,
        model_sha: str = MODEL_SHA,
        gpu_name: str = "NVIDIA GeForce RTX 3090",
        top_p: float = 1,
    ) -> Path:
        run_dir = root / name
        json_dir = run_dir / "json"
        (run_dir / "csv").mkdir(parents=True)
        (run_dir / "text").mkdir()
        json_dir.mkdir()
        config = {
            "runtime": runtime,
            "model": model,
            "context_window": context,
            "runtime_version": f"{runtime}-test-version",
            "cache_policy": f"{runtime}-cache-recorded",
            "server_command": f"{runtime} serve /model",
        }
        model_identity = {
            "source": "local-gguf",
            "requested_model": "shared/model.gguf",
            "resolved_revision": "local-file",
            "sha256": model_sha,
            "bytes": 123456,
        }
        benchmark = {
            "smoke": False,
            "requests": 2,
            "repetitions": 1,
            "warmup_requests_per_case": 1,
            "scenarios": ["short"],
            "input_tokens": [256],
            "output_tokens": 256,
            "mode": "independent",
            "generation_parameters": {
                "temperature": 0,
                "top_p": top_p,
                "max_output_tokens": 256,
                "stream": True,
            },
        }
        argv = [
            runtime,
            "serve",
            "/model",
            "--kv-cache-dtype",
            "fp8",
            "--enable-prefix-caching",
        ]
        runtime_process = {"ownership": "owned-process-group", "argv": argv}
        runtime_environment = {
            "CUDA_VISIBLE_DEVICES": "0",
            "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        }
        cache_policy_details = {
            "declared": "recorded",
            "kv_cache": {
                "state": "enabled-required-for-autoregressive-decode",
                "configuration": {"dtype": "fp8"},
            },
            "effective_cache_argv": [
                "--kv-cache-dtype",
                "fp8",
                "--enable-prefix-caching",
            ],
            "prefix_cache": {
                "state": "enabled-explicit-argv",
                "effective_argv_flags": [
                    "--kv-cache-dtype",
                    "--enable-prefix-caching",
                ],
            },
            "model_residency": {
                "state": "owned-process-started-for-this-run",
                "keep_alive_policy": None,
            },
            "runtime_environment": runtime_environment,
        }
        dataset_manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": status,
            "runtime": runtime,
            "model": model,
            "config": config,
            "model_identity": model_identity,
            "tokenizer": {"sha256": TOKENIZER_SHA, "files": {}},
            "runtime_process": runtime_process,
            "cache_policy_details": cache_policy_details,
            "benchmark": benchmark,
            "environment": {
                "platform": "Linux-test-x86_64",
                "hardware": {
                    "cpu_model": "AMD EPYC test",
                    "cpu_logical_count": 16,
                    "cpu_physical_count": 8,
                    "ram_total_bytes": 64 * 1024**3,
                    "gpu_name": gpu_name,
                    "gpu_memory_total_mib": 24576,
                },
                "runtime_environment": runtime_environment,
            },
            "prometheus": {"metrics_observed": ["custom_runtime_metric"]},
            "files": [],
        }
        final_manifest = {
            "experiment_id": experiment_id,
            "status": status,
            "config": config,
            "model_identity": model_identity,
            "tokenizer": {"sha256": TOKENIZER_SHA, "files": {}},
            "runtime_process": runtime_process,
            "runtime_environment": runtime_environment,
            "cache_policy_details": cache_policy_details,
            "smoke": False,
            "requests": 2,
            "repetitions": 1,
            "warmup_requests_per_case": 1,
            "scenarios": ["short"],
            "seed": 42,
            "profile": "synchronous",
            "mode": "independent",
            "conversation_turns": 1,
            "generation_parameters": benchmark["generation_parameters"],
            "platform": "Linux-test-x86_64",
            "hardware": dataset_manifest["environment"]["hardware"],
            "blocks": [
                {
                    "prefix": "r1-short-measure",
                    "phase": "measure",
                    "scenario": "short",
                    "repetition": 1,
                }
            ],
        }
        measurement_config = {
            "spec": {
                "backend": {
                    "extras": {"body": {"temperature": 0, "top_p": top_p}}
                },
                "profile": {"kind": "synchronous"},
                "data": [{"prompt_tokens": 256, "output_tokens": 256}],
                "seed": {"kind": "static", "value": 42},
            }
        }
        write_json(json_dir / "dataset-manifest.json", dataset_manifest)
        write_json(json_dir / "manifest.json", final_manifest)
        write_json(json_dir / "r1-short-measure-config.json", measurement_config)
        return run_dir

    def write_tables(
        self,
        run_dir: Path,
        *,
        ttfts: tuple[float, ...] = (100.0, 300.0),
        decode_rates: tuple[float, ...] = (10.0, 20.0),
        include_warmup: bool = False,
        include_gpu: bool = False,
    ) -> None:
        manifest = json.loads((run_dir / "json" / "dataset-manifest.json").read_text())
        experiment_id = manifest["experiment_id"]
        runtime = manifest["runtime"]
        model = manifest["model"]
        request_rows = []
        for index, (ttft, decode) in enumerate(zip(ttfts, decode_rates), 1):
            request_rows.append(
                {
                    "experiment_id": experiment_id,
                    "request_uid": f"{experiment_id}:r{index}",
                    "runtime": runtime,
                    "model": model,
                    "block": "r1-short-measure",
                    "benchmark_phase": "measure",
                    "scenario": "short",
                    "status": "successful",
                    "time_to_first_token_ms": ttft,
                    "decode_tokens_per_second": decode,
                    "end_to_end_latency_seconds": 1.0,
                }
            )
        if include_warmup:
            request_rows.append(
                {
                    "experiment_id": experiment_id,
                    "request_uid": f"{experiment_id}:warmup",
                    "runtime": runtime,
                    "model": model,
                    "block": "r1-short-warmup",
                    "benchmark_phase": "warmup",
                    "scenario": "short",
                    "status": "successful",
                    "time_to_first_token_ms": 999.0,
                    "decode_tokens_per_second": 1.0,
                    "end_to_end_latency_seconds": 2.0,
                }
            )
        pd.DataFrame(request_rows).to_csv(run_dir / "csv" / "all-requests.csv", index=False)

        request_uid = request_rows[0]["request_uid"]
        pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "request_uid": request_uid,
                    "event": "request_start",
                    "timestamp_s": 1.0,
                    "request_elapsed_s": 0.0,
                },
                {
                    "experiment_id": experiment_id,
                    "request_uid": request_uid,
                    "event": "first_content",
                    "timestamp_s": 1.1,
                    "request_elapsed_s": 0.1,
                },
            ]
        ).to_csv(run_dir / "csv" / "request-events.csv", index=False)

        metrics = [("inference_host_cpu_utilization_ratio", 0.5, "fraction")]
        if include_gpu:
            metrics.append(("inference_gpu_utilization_ratio", 0.8, "fraction"))
        samples = []
        telemetry = []
        phase_metrics = []
        for offset, (metric, value, unit) in enumerate(metrics):
            samples.append(
                {
                    "experiment_id": experiment_id,
                    "timestamp_s": 1.0 + offset / 10,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "kind": "gauge",
                }
            )
            telemetry.append(
                {
                    "experiment_id": experiment_id,
                    "request_uid": request_uid,
                    "runtime": runtime,
                    "model": model,
                    "benchmark_phase": "measure",
                    "observed_phase": "prefill_observed",
                    "request_elapsed_s": offset / 10,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                    "kind": "gauge",
                }
            )
            phase_metrics.append(
                {
                    "experiment_id": experiment_id,
                    "request_uid": request_uid,
                    "runtime": runtime,
                    "model": model,
                    "benchmark_phase": "measure",
                    "observed_phase": "prefill_observed",
                    "metric": metric,
                    "unit": unit,
                    "kind": "gauge",
                    "sample_count": 1,
                    "mean": value,
                    "p95": value,
                }
            )
        pd.DataFrame(samples).to_csv(run_dir / "csv" / "prometheus-samples.csv", index=False)
        pd.DataFrame(telemetry).to_csv(
            run_dir / "csv" / "telemetry-by-request.csv", index=False
        )
        pd.DataFrame(phase_metrics).to_csv(
            run_dir / "csv" / "request-phase-metrics.csv", index=False
        )

    def finalize_run(self, run_dir: Path) -> None:
        dataset_path = run_dir / "json" / "dataset-manifest.json"
        final_path = run_dir / "json" / "manifest.json"
        manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
        records = []
        for filename in TABLE_NAMES:
            path = run_dir / "csv" / filename
            if not path.is_file():
                continue
            records.append(
                {
                    "name": filename.removesuffix(".csv").replace("-", "_"),
                    "path": f"csv/{filename}",
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        manifest["files"] = records
        write_json(dataset_path, manifest)
        checksum_rows = records + [
            {"path": "json/dataset-manifest.json", "sha256": file_sha256(dataset_path)},
            {"path": "json/manifest.json", "sha256": file_sha256(final_path)},
        ]
        (run_dir / "text" / "SHA256SUMS").write_text(
            "".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows),
            encoding="utf-8",
        )

    def valid_run(self, root: Path, name: str, **kwargs) -> Path:
        run = self.write_run(root, name, **kwargs)
        self.write_tables(run)
        self.finalize_run(run)
        return run

    def test_builds_integrity_checked_cross_runtime_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "downloaded"
            output = root / "dataset"

            first = self.write_run(
                results,
                "vllm-run",
                status="complete",
                experiment_id="exp-vllm",
                runtime="vllm",
            )
            self.write_tables(first, include_warmup=True)
            self.finalize_run(first)

            second = self.write_run(
                results,
                "ollama-run",
                status="complete",
                experiment_id="exp-ollama",
                runtime="ollama",
            )
            self.write_tables(
                second,
                ttfts=(400.0,),
                decode_rates=(8.0,),
                include_gpu=True,
            )
            self.finalize_run(second)

            partial = self.write_run(
                results,
                "failed-run",
                status="failed",
                experiment_id="exp-failed",
                runtime="vllm",
            )
            self.write_tables(partial, ttfts=(1.0,), decode_rates=(1000.0,))
            self.finalize_run(partial)

            manifest = build_observability_dataset(results, output, write_parquet=True)

            self.assertEqual(manifest["runs_discovered"], 3)
            self.assertEqual(manifest["runs_complete"], 2)
            self.assertEqual(manifest["runs_comparable"], 2)
            self.assertEqual(manifest["runs_excluded"], 1)
            self.assertEqual(manifest["comparison_groups"], 1)

            inventory = pd.read_csv(output / "run-inventory.csv")
            comparable = inventory.loc[inventory["included_in_comparison"]]
            self.assertEqual(comparable["comparison_id"].nunique(), 1)
            self.assertEqual(set(comparable["runtime"]), {"vllm", "ollama"})
            self.assertTrue(comparable["integrity_ok"].all())
            self.assertNotIn("runtime_server_command", inventory.columns)
            vllm = comparable.loc[comparable["runtime"] == "vllm"].iloc[0]
            self.assertEqual(
                json.loads(vllm["runtime_process_argv_json"]),
                [
                    "vllm",
                    "serve",
                    "/model",
                    "--kv-cache-dtype",
                    "fp8",
                    "--enable-prefix-caching",
                ],
            )
            self.assertRegex(vllm["runtime_process_argv_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                json.loads(vllm["runtime_environment_json"]),
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
                },
            )
            self.assertRegex(vllm["runtime_environment_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                json.loads(vllm["effective_cache_flags_json"]),
                [
                    {"flag": "--kv-cache-dtype", "value": "fp8"},
                    {"flag": "--enable-prefix-caching", "value": True},
                ],
            )
            self.assertEqual(vllm["prefix_cache_state"], "enabled-explicit-argv")
            self.assertEqual(
                json.loads(vllm["effective_cache_argv_json"]),
                ["--kv-cache-dtype", "fp8", "--enable-prefix-caching"],
            )
            self.assertEqual(
                json.loads(vllm["kv_cache_configuration_json"]), {"dtype": "fp8"}
            )
            self.assertEqual(
                vllm["kv_cache_state"],
                "enabled-required-for-autoregressive-decode",
            )
            self.assertEqual(
                vllm["model_residency"], "owned-process-started-for-this-run"
            )
            identity = json.loads(vllm["comparison_identity_json"])
            self.assertEqual(
                identity["config"]["generation_parameters_source"], "manifest"
            )
            failed = inventory.loc[inventory["run_id"] == "exp-failed"].iloc[0]
            self.assertFalse(bool(failed["included_in_comparison"]))

            groups = pd.read_csv(output / "comparison-groups.csv")
            self.assertEqual(groups.loc[0, "runtime_count"], 2)
            self.assertTrue(bool(groups.loc[0, "cross_runtime_ready"]))

            requests = pd.read_csv(output / "all-requests.csv")
            self.assertEqual(set(requests["source_run_id"]), {"exp-vllm", "exp-ollama"})
            self.assertEqual(requests["comparison_id"].nunique(), 1)

            summary = pd.read_csv(output / "experiment-summary.csv")
            measure = summary.loc[
                (summary["run_id"] == "exp-vllm") & (summary["phase"] == "measure")
            ].iloc[0]
            warmup = summary.loc[
                (summary["run_id"] == "exp-vllm") & (summary["phase"] == "warmup")
            ].iloc[0]
            self.assertEqual(measure["ttft_ms_p50"], 200.0)
            self.assertEqual(measure["decode_tokens_s_mean"], 15.0)
            self.assertEqual(measure["prefix_cache_state"], "enabled-explicit-argv")
            self.assertEqual(
                json.loads(measure["effective_cache_flags_json"])[0],
                {"flag": "--kv-cache-dtype", "value": "fp8"},
            )
            self.assertEqual(warmup["ttft_ms_p50"], 999.0)

            coverage = pd.read_csv(output / "metric-coverage.csv")
            vllm_gpu = coverage.loc[
                (coverage["run_id"] == "exp-vllm")
                & (coverage["metric"] == "inference_gpu_utilization_ratio")
            ].iloc[0]
            ollama_gpu = coverage.loc[
                (coverage["run_id"] == "exp-ollama")
                & (coverage["metric"] == "inference_gpu_utilization_ratio")
            ].iloc[0]
            self.assertFalse(bool(vllm_gpu["available"]))
            self.assertTrue(pd.isna(vllm_gpu["mean"]))
            self.assertEqual(vllm_gpu["sample_count"], 0)
            self.assertTrue(bool(ollama_gpu["available"]))
            self.assertEqual(ollama_gpu["mean"], 0.8)

    def test_control_change_creates_another_comparison_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.valid_run(
                root,
                "vllm-run",
                status="complete",
                experiment_id="exp-vllm",
                runtime="vllm",
                context=8192,
            )
            second = self.valid_run(
                root,
                "ollama-run",
                status="complete",
                experiment_id="exp-ollama",
                runtime="ollama",
                context=4096,
            )
            self.assertTrue(first.is_dir() and second.is_dir())

            output = root / "out"
            build_observability_dataset(root, output)

            inventory = pd.read_csv(output / "run-inventory.csv")
            self.assertEqual(inventory["comparison_id"].nunique(), 2)
            groups = pd.read_csv(output / "comparison-groups.csv")
            self.assertEqual(len(groups), 2)
            self.assertFalse(groups["cross_runtime_ready"].any())

    def test_generation_parameters_are_part_of_the_comparison_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.valid_run(
                root,
                "vllm-run",
                status="complete",
                experiment_id="exp-vllm",
                runtime="vllm",
                top_p=1,
            )
            self.valid_run(
                root,
                "ollama-run",
                status="complete",
                experiment_id="exp-ollama",
                runtime="ollama",
                top_p=0.9,
            )

            output = root / "out"
            build_observability_dataset(root, output)

            inventory = pd.read_csv(output / "run-inventory.csv")
            self.assertEqual(inventory["comparison_id"].nunique(), 2)

    def test_existing_server_does_not_turn_unknown_argv_into_empty_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.write_run(
                root,
                "existing-server",
                status="complete",
                experiment_id="exp-existing",
                runtime="ollama",
            )
            for name in ("dataset-manifest.json", "manifest.json"):
                path = run / "json" / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["runtime_process"] = {
                    "ownership": "existing-server",
                    "argv": None,
                }
                payload["cache_policy_details"] = {
                    "kv_cache": {
                        "state": "enabled-required-for-autoregressive-decode"
                    },
                    "prefix_cache": {
                        "state": "not-exposed-by-owned-launch-argv",
                        "effective_argv_flags": [],
                    },
                    "model_residency": "existing-server-state-unknown",
                }
                write_json(path, payload)
            self.write_tables(run)
            self.finalize_run(run)

            output = root / "out"
            build_observability_dataset(root, output)
            inventory = pd.read_csv(output / "run-inventory.csv")
            self.assertTrue(pd.isna(inventory.loc[0, "runtime_process_argv_json"]))
            self.assertTrue(
                pd.isna(inventory.loc[0, "runtime_process_argv_fingerprint"])
            )
            self.assertTrue(pd.isna(inventory.loc[0, "effective_cache_flags_json"]))
            self.assertIn("argv is unavailable", inventory.loc[0, "compatibility_reason"])

    def test_failed_gpu_inventory_is_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.valid_run(
                root,
                "valid-gpu",
                status="complete",
                experiment_id="exp-valid-gpu",
                runtime="vllm",
            )
            failed = self.write_run(
                root,
                "failed-gpu",
                status="complete",
                experiment_id="exp-failed-gpu",
                runtime="ollama",
            )
            for name in ("dataset-manifest.json", "manifest.json"):
                path = failed / "json" / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                hardware = (
                    payload["environment"]["hardware"]
                    if name == "dataset-manifest.json"
                    else payload["hardware"]
                )
                hardware["gpu_inventory"] = {
                    "returncode": 1,
                    "stdout": "",
                    "stderr": "NVIDIA-SMI has failed",
                }
                write_json(path, payload)
            self.write_tables(failed)
            self.finalize_run(failed)

            output = root / "out"
            build_observability_dataset(root, output)
            inventory = pd.read_csv(output / "run-inventory.csv")
            failed_row = inventory.loc[inventory["run_id"] == "exp-failed-gpu"].iloc[0]
            self.assertIn("hardware.gpu", failed_row["comparison_identity_missing"])
            self.assertTrue(bool(failed_row["included_in_comparison"]))
            groups = pd.read_csv(output / "comparison-groups.csv")
            self.assertEqual(len(groups), 2)
            self.assertFalse(groups["cross_runtime_ready"].any())

    def test_malformed_or_hash_invalid_artifacts_stay_only_in_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            malformed = self.write_run(
                root,
                "malformed",
                status="complete",
                experiment_id="exp-malformed",
                runtime="llama",
            )
            self.write_tables(malformed)
            (malformed / "csv" / "all-requests.csv").write_text(
                '"unterminated\n', encoding="utf-8"
            )
            self.finalize_run(malformed)

            corrupt = self.valid_run(
                root,
                "corrupt",
                status="complete",
                experiment_id="exp-corrupt",
                runtime="vllm",
            )
            with (corrupt / "csv" / "prometheus-samples.csv").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write("exp-corrupt,2.0,metric,1,fraction,gauge\n")

            missing = self.write_run(
                root,
                "missing",
                status="complete",
                experiment_id="exp-missing",
                runtime="ollama",
            )
            self.write_tables(missing)
            (missing / "csv" / "telemetry-by-request.csv").unlink()
            self.finalize_run(missing)

            output = root / "out"
            manifest = build_observability_dataset(root, output)

            self.assertEqual(manifest["runs_complete"], 3)
            self.assertEqual(manifest["runs_comparable"], 0)
            inventory = pd.read_csv(output / "run-inventory.csv")
            self.assertFalse(inventory["included_in_comparison"].any())
            errors = "\n".join(inventory["integrity_errors"].fillna(""))
            self.assertIn("CSV malformado", errors)
            self.assertIn("SHA-256 divergente", errors)
            self.assertIn("telemetry-by-request.csv", errors)
            self.assertTrue(pd.read_csv(output / "all-requests.csv").empty)
            self.assertTrue(pd.read_csv(output / "complete-runs.csv").empty)

    def test_final_status_and_manifest_hash_are_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self.write_run(
                root,
                "cleanup-failed",
                status="complete",
                experiment_id="exp-cleanup-failed",
                runtime="vllm",
            )
            self.write_tables(run)
            final_path = run / "json" / "manifest.json"
            final = json.loads(final_path.read_text())
            final["status"] = "failed"
            final["error"] = "server cleanup failed"
            write_json(final_path, final)
            self.finalize_run(run)

            output = root / "out"
            build_observability_dataset(root, output)
            inventory = pd.read_csv(output / "run-inventory.csv")
            self.assertEqual(inventory.loc[0, "status"], "failed")
            self.assertFalse(bool(inventory.loc[0, "included_in_comparison"]))

            # A mesma run continua excluida se o manifesto final for alterado
            # depois do SHA256SUMS.
            final["status"] = "complete"
            write_json(final_path, final)
            build_observability_dataset(root, root / "out-tampered")
            tampered = pd.read_csv(root / "out-tampered" / "run-inventory.csv")
            self.assertIn("json/manifest.json", tampered.loc[0, "integrity_errors"])

    def test_requires_at_least_one_supported_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(discover_runs(root), [])
            with self.assertRaisesRegex(ValueError, "Nenhum json"):
                build_observability_dataset(root, root / "out")

    def test_notebook_is_offline_and_all_code_cells_compile(self):
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "analyze_runtime_observability.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        sources = []
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            sources.append(source)
            if cell["cell_type"] == "code":
                compile(source, f"{notebook_path.name}:{cell['id']}", "exec")
        content = "\n".join(sources)
        self.assertIn("OBSERVABILITY_DATASET_DIR", content)
        self.assertIn("comparison_id", content)
        self.assertNotIn("http://", content)
        self.assertNotIn("https://", content)
        self.assertNotIn("import requests", content)

    def test_notebook_executes_against_a_local_consolidated_dataset(self):
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "analyze_runtime_observability.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict("os.environ", {"MPLCONFIGDIR": str(root / "matplotlib")}):
                import matplotlib

                matplotlib.use("Agg")
            self.valid_run(
                root / "results",
                "complete-run",
                status="complete",
                experiment_id="exp-local",
                runtime="vllm",
            )
            dataset_dir = root / "dataset"
            build_observability_dataset(root / "results", dataset_dir)

            namespace = {"__name__": "observability_notebook_test"}
            environment = {
                "OBSERVABILITY_DATASET_DIR": str(dataset_dir),
                "OBSERVABILITY_RUN_ID_VLLM": "exp-local",
                "OBSERVABILITY_RUN_ID_LLAMA": "",
                "OBSERVABILITY_RUN_ID_OLLAMA": "",
                "MPLCONFIGDIR": str(root / "matplotlib"),
            }
            with warnings.catch_warnings(), patch.dict("os.environ", environment), patch(
                "IPython.display.display", lambda *_args, **_kwargs: None
            ), patch("builtins.print"):
                warnings.simplefilter("ignore")
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec("".join(cell["source"]), namespace)
            self.assertTrue(namespace["vllm"]["ready"])
            self.assertEqual(namespace["delivery_vllm"].loc[0, "total"], 2)
            self.assertEqual(namespace["delivery_vllm"].loc[0, "ttft_p50_ms"], 200.0)
            self.assertFalse(namespace["comparison_gate"]["comparable"])

            labels = {"vllm": "vLLM", "llama": "llama.cpp", "ollama": "Ollama"}
            runtime_results = {}
            for runtime_key, runtime_label in labels.items():
                actual_runtime = "llama.cpp" if runtime_key == "llama" else runtime_key
                run_id = f"run-{runtime_key}"
                requests = pd.DataFrame(
                    [
                        {
                            "source_run_id": run_id,
                            "request_uid": f"{run_id}:{index}",
                            "benchmark_phase": "measure",
                            "scenario": scenario,
                            "status": "successful",
                            "request_sha256": f"sha-{scenario}",
                            "time_to_first_token_ms": 100.0,
                            "decode_tokens_per_second": 20.0,
                            "end_to_end_latency_seconds": 1.0,
                        }
                        for index, scenario in enumerate(["short", "medium", "long"], 1)
                    ]
                )
                runtime_results[runtime_key] = {
                    "runtime_key": runtime_key,
                    "label": runtime_label,
                    "run_id": run_id,
                    "configured": True,
                    "found": True,
                    "ready": True,
                    "issues": [],
                    "identity": pd.Series(
                        {
                            "runtime": actual_runtime,
                            "comparison_id": "comparison-123",
                            "comparison_identity_complete": True,
                            "model_sha256": MODEL_SHA,
                            "tokenizer_sha256": TOKENIZER_SHA,
                            "context_window": 8192,
                            "output_tokens": 256,
                        }
                    ),
                    "requests": requests,
                }
            namespace["tables"]["groups"] = pd.DataFrame(
                [{"comparison_id": "comparison-123", "runtime_count": 3, "cross_runtime_ready": True}]
            )
            synthetic_gate = namespace["build_comparability"](runtime_results)
            self.assertTrue(synthetic_gate["comparable"])
            self.assertTrue(synthetic_gate["pairing_verified"])

            performance_rows = []
            resource_rows = []
            energy_rows = []
            for runtime_index, (runtime_key, runtime_label) in enumerate(labels.items()):
                for scenario_index, scenario in enumerate(["short", "medium", "long"]):
                    performance_rows.append(
                        {
                            "runtime_key": runtime_key,
                            "runtime": runtime_label,
                            "scenario": scenario,
                            "total": 2,
                            "successful": 2,
                            "failed": 0,
                            "success_rate_pct": 100.0,
                            "prompt_tokens_p50": 256 * (scenario_index + 1),
                            "completion_tokens_p50": 128.0,
                            "ttft_n": 2,
                            "ttft_p50_ms": 80 + 20 * runtime_index + 100 * scenario_index,
                            "ttft_p95_ms": 100 + 20 * runtime_index + 100 * scenario_index,
                            "decode_n": 2,
                            "decode_p05_tps": 30 - runtime_index - scenario_index,
                            "decode_p50_tps": 32 - runtime_index - scenario_index,
                            "decode_p95_tps": 34 - runtime_index - scenario_index,
                            "e2e_n": 2,
                            "e2e_p50_s": 1 + runtime_index / 10 + scenario_index,
                            "e2e_p95_s": 1.2 + runtime_index / 10 + scenario_index,
                        }
                    )
                    for metric, median, peak in [
                        ("inference_gpu_utilization_ratio", 70 + runtime_index, 90 + runtime_index),
                        ("inference_gpu_memory_used_bytes", 10 + runtime_index, 12 + runtime_index),
                        ("inference_host_cpu_utilization_ratio", 20 + runtime_index, 30 + runtime_index),
                        ("inference_host_memory_used_bytes", 24 + runtime_index, 28 + runtime_index),
                    ]:
                        resource_rows.append(
                            {
                                "runtime": runtime_label,
                                "scenario": scenario,
                                "observed_phase": "decode_observed",
                                "metric": metric,
                                "median": median,
                                "peak": peak,
                                "coverage": 1.0,
                            }
                        )
                    for request_index in range(2):
                        energy_rows.append(
                            {
                                "runtime": runtime_label,
                                "scenario": scenario,
                                "request_uid": f"{runtime_key}:{scenario}:{request_index}",
                                "energy_j": 100 + 10 * runtime_index + request_index,
                                "energy_j_per_token": 1 + runtime_index / 10,
                            }
                        )

            performance = pd.DataFrame(performance_rows)
            resources = pd.DataFrame(resource_rows)
            energy = pd.DataFrame(energy_rows)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertIn("Resultado relativo", namespace["result_heatmap"](performance).to_html())
                self.assertIn("Uso e pressão", namespace["resource_heatmap"](resources, performance).to_html())
                self.assertEqual(len(namespace["slo_summary"](performance)), 3)
                self.assertIn("Decisão", namespace["comparison_findings"](performance, resources, energy))
                namespace["plot_comparison_performance"](performance)
                namespace["plot_comparison_resources"](resources)
                self.assertTrue(namespace["plot_energy_comparison"](energy, performance))
            namespace["plt"].close("all")


if __name__ == "__main__":
    unittest.main()
