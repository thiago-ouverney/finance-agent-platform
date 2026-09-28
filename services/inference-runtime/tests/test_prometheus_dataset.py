import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prometheus_dataset as subject
from results_layout import prepare
from scripts.verify_observability_results import verify_tree


class CorrelationTests(unittest.TestCase):
    def request(self):
        return {
            "experiment_id": "exp-1",
            "request_uid": "request-1",
            "runtime": "vllm",
            "model": "model",
            "block": "r1-short-measure",
            "benchmark_phase": "measure",
            "scenario": "short",
            "repetition": 1,
            "request_start_epoch_s": 10.0,
            "first_content_epoch_s": 11.0,
            "last_content_epoch_s": 13.0,
            "request_end_epoch_s": 14.0,
        }

    def test_boundaries_are_explicit(self):
        request = self.request()
        self.assertEqual(subject.observed_phase(request, 10.9), "prefill_observed")
        self.assertEqual(subject.observed_phase(request, 11.0), "decode_observed")
        self.assertEqual(subject.observed_phase(request, 13.0), "decode_observed")
        self.assertEqual(subject.observed_phase(request, 13.1), "response_close")

    def test_missing_first_content_is_not_reported_as_prefill_or_zero(self):
        request = self.request()
        request["first_content_epoch_s"] = None
        request["last_content_epoch_s"] = None
        self.assertEqual(subject.observed_phase(request, 12.0), "request_without_first_content")

    def test_prometheus_url_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(ValueError, "sem credenciais"):
            subject.normalize_prometheus_url("http://user:secret@localhost:9090")

    def test_counters_use_delta_rate_and_gauges_use_distributions(self):
        samples = [
            {"timestamp_s": 10.5, "metric": "inference_host_cpu_utilization_ratio", "value": .2, "labels": {}},
            {"timestamp_s": 11.5, "metric": "inference_host_cpu_utilization_ratio", "value": .6, "labels": {}},
            {"timestamp_s": 11.0, "metric": "inference_host_disk_read_bytes_total", "value": 100, "labels": {}},
            {"timestamp_s": 13.0, "metric": "inference_host_disk_read_bytes_total", "value": 300, "labels": {}},
        ]
        rows = subject.correlate_samples(samples, [self.request()], 9.0)
        summary = subject.phase_metrics(rows)
        cpu = next(row for row in summary if row["metric"].endswith("cpu_utilization_ratio")
                   and row["observed_phase"] == "prefill_observed")
        disk = next(row for row in summary if row["metric"].endswith("disk_read_bytes_total"))
        self.assertEqual(cpu["mean"], .2)
        self.assertIsNone(disk["mean"])
        self.assertEqual(disk["delta"], 200)
        self.assertEqual(disk["rate_mean"], 100)


class ExportTests(unittest.TestCase):
    def test_export_writes_portable_dataset_and_checksums(self):
        request = CorrelationTests().request()
        request.update({
            "status": "successful",
            "request_index": 1,
            "request_sha256": "abc",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "output": "conteúdo privado",
            "request_args": "{\"body\": {\"messages\": [\"privado\"]}}",
        })
        manifest = {
            "experiment_id": "exp-1",
            "status": "complete",
            "config": {"runtime": "vllm", "model": "model"},
            "tokenizer": {"sha256": "tokenizer-hash"},
            "runtime_process": {"argv": ["vllm", "serve"]},
            "cache_policy_details": {"prefix_cache": {"state": "disabled"}},
            "hardware": {"cpu_model": "test-cpu", "ram_total_bytes": 1024},
        }
        samples = [
            {"timestamp_s": 10.5, "metric": "inference_host_cpu_utilization_ratio", "value": .2,
             "labels": {"job": "inference-host", "instance": "127.0.0.1:9108"}},
            {"timestamp_s": 11.5, "metric": "inference_gpu_memory_used_bytes", "value": 1024,
             "labels": {"gpu": "0"}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = prepare(Path(temporary))
            (output / "json" / "manifest.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            with mock.patch.object(subject, "prometheus_ready", return_value=True), \
                 mock.patch.object(subject, "query_range", return_value=samples), \
                 mock.patch.object(subject.time, "time", return_value=20.0):
                result = subject.export_dataset(
                    output=output,
                    manifest=manifest,
                    requests=[request],
                    prometheus_url="http://127.0.0.1:9090",
                    run_start_epoch_s=9.0,
                    run_end_epoch_s=15.0,
                    margin_seconds=0,
                )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["request_count"], 1)
            for filename in (
                "all-requests.csv", "request-events.csv", "prometheus-samples.csv",
                "telemetry-by-request.csv", "request-phase-metrics.csv",
            ):
                self.assertTrue((output / "csv" / filename).is_file(), filename)
            dataset = json.loads((output / "json" / "dataset-manifest.json").read_text())
            self.assertEqual(dataset["prometheus"]["metrics_observed"], [
                "inference_gpu_memory_used_bytes", "inference_host_cpu_utilization_ratio",
            ])
            self.assertEqual(dataset["tokenizer"]["sha256"], "tokenizer-hash")
            self.assertEqual(dataset["runtime_process"]["argv"], ["vllm", "serve"])
            self.assertEqual(dataset["environment"]["hardware"]["cpu_model"], "test-cpu")
            with (output / "csv" / "telemetry-by-request.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["observed_phase"] for row in rows],
                             ["prefill_observed", "decode_observed"])
            request_header = (output / "csv" / "all-requests.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("output", request_header)
            self.assertNotIn("request_args", request_header)
            self.assertTrue((output / "text" / "SHA256SUMS").is_file())
            count, errors = verify_tree(output)
            self.assertEqual(count, 1)
            self.assertEqual(errors, [])

            final_manifest_path = output / "json" / "manifest.json"
            final_manifest_path.write_text('{"status":"adulterado"}\n', encoding="utf-8")
            _count, errors = verify_tree(output)
            self.assertTrue(any(
                "SHA-256 divergente: json/manifest.json" in error
                for error in errors
            ))
            final_manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

            dataset_path = output / "json" / "dataset-manifest.json"
            original_dataset = dataset_path.read_bytes()
            dataset_path.write_text('{"status":"adulterado"}\n', encoding="utf-8")
            _count, errors = verify_tree(output)
            self.assertTrue(any(
                "SHA-256 divergente: json/dataset-manifest.json" in error
                for error in errors
            ))
            dataset_path.write_bytes(original_dataset)

            requests_path = output / "csv" / "all-requests.csv"
            original_requests = requests_path.read_bytes()
            requests_path.write_text("alterado", encoding="utf-8")
            _count, errors = verify_tree(output)
            self.assertTrue(any("SHA-256 divergente" in error for error in errors))
            requests_path.write_bytes(original_requests)

            checksum_path = output / "text" / "SHA256SUMS"
            original_checksums = checksum_path.read_text(encoding="utf-8")
            checksum_path.write_text(
                "".join(
                    line + "\n" for line in original_checksums.splitlines()
                    if not line.endswith("csv/request-events.csv")
                ),
                encoding="utf-8",
            )
            _count, errors = verify_tree(output)
            self.assertTrue(any(
                "entrada estrutural ausente: csv/request-events.csv" in error
                for error in errors
            ))
            checksum_path.write_text(original_checksums, encoding="utf-8")

            checksum_path.unlink()
            _count, errors = verify_tree(output)
            self.assertTrue(any("SHA256SUMS: arquivo estrutural ausente" in error for error in errors))

    def test_empty_prometheus_window_fails_instead_of_writing_zeros(self):
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(subject, "prometheus_ready", return_value=True), \
             mock.patch.object(subject, "query_range", return_value=[]), \
             mock.patch.object(subject.time, "time", return_value=20.0):
            with self.assertRaisesRegex(RuntimeError, "não retornou amostras"):
                subject.export_dataset(
                    output=prepare(Path(temporary)),
                    manifest={"experiment_id": "x", "status": "failed", "config": {}},
                    requests=[], prometheus_url="http://127.0.0.1:9090",
                    run_start_epoch_s=9.0, run_end_epoch_s=15.0, margin_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
