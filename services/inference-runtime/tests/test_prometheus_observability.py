import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prometheus_profile import plot_profile, profile_request
from scripts.host_metrics_exporter import collect_metrics, read_vmstat
from scripts.prometheus_stack import assert_port_available, render_config, wait_owned_url


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def send_payload(self, payload, content_type="application/json"):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/-/ready":
            self.send_payload(b"ready\n", "text/plain")
            return
        if self.path == "/v1/models":
            self.send_payload({"data": [{"id": "mock-model"}]})
            return
        if self.path.startswith("/api/v1/query_range"):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            start = float(query["start"][0])
            end = float(query["end"][0])
            self.send_payload({
                "status": "success",
                "data": {"resultType": "matrix", "result": [{
                    "metric": {"__name__": "inference_host_cpu_utilization_ratio"},
                    "values": [[start, "0.2"], [end, "0.4"]],
                }]},
            })
            return
        self.send_error(404)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for payload in (
            {"choices": [{"delta": {"content": "Resposta"}}]},
            {"choices": [{"delta": {"content": " sequencial"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        ):
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class PrometheusProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_profile_request_correlates_events_and_telemetry(self):
        result = profile_request(
            base_url=self.url,
            model="mock-model",
            messages=[{"role": "user", "content": "teste"}],
            prometheus_url=self.url,
            max_tokens=4,
            margin_before_seconds=0,
            margin_after_seconds=0,
        )
        self.assertEqual(result["request"]["completion_tokens"], 2)
        self.assertEqual(result["request"]["output"], "Resposta sequencial")
        self.assertEqual([event["name"] for event in result["events"]],
                         ["request_start", "first_token", "last_content", "request_end"])
        self.assertEqual({row["metric"] for row in result["telemetry"]},
                         {"inference_host_cpu_utilization_ratio"})
        self.assertGreaterEqual(result["events"][-1]["elapsed_seconds"],
                                result["events"][1]["elapsed_seconds"])

    def test_plot_keeps_phase_markers_when_telemetry_is_empty(self):
        profile = {
            "metadata": {"model": "mock-model"},
            "request": {
                "time_to_first_token_ms": 100,
                "decode_tokens_per_second": 20,
            },
            "events": [
                {"name": "request_start", "timestamp": 10.0, "elapsed_seconds": 0.0},
                {"name": "first_token", "timestamp": 10.1, "elapsed_seconds": 0.1},
                {"name": "last_content", "timestamp": 11.0, "elapsed_seconds": 1.0},
                {"name": "request_end", "timestamp": 11.1, "elapsed_seconds": 1.1},
            ],
            "telemetry": [],
        }
        figure = plot_profile(profile)
        self.assertEqual(len(figure.axes), 5)


class HostExporterTests(unittest.TestCase):
    def test_vmstat_and_missing_gpu_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "vmstat"
            path.write_text("pgfault 123\npgmajfault 4\n", encoding="utf-8")
            self.assertEqual(read_vmstat(path), {"pgfault": 123, "pgmajfault": 4})
        with mock.patch("scripts.host_metrics_exporter.query_gpus", return_value=(False, [])):
            payload = collect_metrics("missing-nvidia-smi")
        self.assertIn("inference_host_cpu_utilization_ratio", payload)
        self.assertIn("inference_host_swap_used_bytes", payload)
        self.assertIn("inference_gpu_scrape_success 0.0", payload)


class StackConfigTests(unittest.TestCase):
    def test_runtime_target_is_optional(self):
        host_only = render_config("127.0.0.1:9108", "500ms")
        self.assertIn("inference-host", host_only)
        self.assertIn("scrape_timeout: 450ms", host_only)
        self.assertNotIn("inference-runtime", host_only)
        with_runtime = render_config("127.0.0.1:9108", "500ms", "127.0.0.1:8000")
        self.assertIn("inference-runtime", with_runtime)
        self.assertIn('targets: ["127.0.0.1:8000"]', with_runtime)

    def test_port_preflight_rejects_an_existing_service(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        try:
            with self.assertRaisesRegex(RuntimeError, "indisponível"):
                assert_port_available("127.0.0.1", server.server_port, "Prometheus")
        finally:
            server.server_close()

    def test_readiness_rejects_unrelated_server_when_child_exited(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                pid_path = Path(temporary) / "prometheus.pid"
                pid_path.write_text("12345\n", encoding="utf-8")
                process = mock.Mock(pid=12345)
                process.poll.return_value = 1
                with self.assertRaisesRegex(RuntimeError, "encerrou antes do readiness"):
                    wait_owned_url(
                        f"http://127.0.0.1:{server.server_port}/-/ready",
                        process,
                        pid_path,
                        "Prometheus",
                        timeout=0.2,
                    )
        finally:
            server.shutdown()
            server.server_close()

    def test_readiness_rejects_pid_file_that_no_longer_names_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "exporter.pid"
            pid_path.write_text("999\n", encoding="utf-8")
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            with self.assertRaisesRegex(RuntimeError, "PID de Exporter mudou"):
                wait_owned_url(
                    "http://127.0.0.1:1/-/healthy",
                    process,
                    pid_path,
                    "Exporter",
                    timeout=0.2,
                )


class ObservabilityMakeTests(unittest.TestCase):
    def make_dry_run(self, target: str, *variables: str) -> str:
        result = subprocess.run(
            ["make", "-n", target, *variables],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def test_runtime_commands_are_selected_without_changing_notebook_code(self):
        vllm = self.make_dry_run("observe-serve-vllm", "OBS_CONTEXT=4096")
        self.assertIn('serve "$model"', vllm)
        self.assertIn('--max-model-len "4096"', vllm)

        llama = self.make_dry_run("observe-serve-llama")
        self.assertIn("llama-server", llama)
        self.assertIn("--parallel 1", llama)

        ollama = self.make_dry_run("observe-serve-ollama")
        self.assertIn("OLLAMA_MODELS=", ollama)
        self.assertIn("serve", ollama)

    def test_vllm_prometheus_includes_runtime_metrics(self):
        output = self.make_dry_run("observe-prometheus-start", "OBS_RUNTIME=vllm")
        self.assertIn('PROMETHEUS_RUNTIME_TARGET="127.0.0.1:8000"', output)

    def test_headless_bench_prepares_then_runs_offline_with_prometheus(self):
        output = self.make_dry_run(
            "observe-bench",
            "OBS_RUNTIME=vllm",
            "OBS_MODEL_SOURCE=hf",
            "OBS_MODEL=Qwen/Qwen2.5-7B-Instruct",
            "OBS_BENCH_REQUESTS=2",
        )
        self.assertIn("prepare_observe_model.py prepare", output)
        self.assertIn("render_observe_runtime.py", output)
        self.assertIn("--runtime-version", output)
        self.assertIn('uv pip install "vllm==0.29.0"', output)
        self.assertIn("HF_HUB_OFFLINE=1", output)
        self.assertIn("--prometheus-url", output)
        self.assertIn('--requests "2"', output)
        self.assertIn("--disable-native-monitor", output)

    def test_vllm_install_fails_early_when_ninja_is_missing(self):
        output = self.make_dry_run("observe-install-vllm")
        self.assertIn("command -v ninja", output)
        self.assertIn("apt-get install -y ninja-build", output)

    def test_vllm_uses_bundled_cuda_runtime_for_import_and_launch(self):
        install = self.make_dry_run("observe-install-vllm")
        serve = self.make_dry_run("observe-serve-vllm")
        bench = self.make_dry_run(
            "observe-bench",
            "OBS_RUNTIME=vllm",
            "OBS_MODEL_SOURCE=hf",
            "OBS_MODEL=Qwen/Qwen2.5-7B-Instruct",
        )
        for output in (install, serve, bench):
            self.assertIn("nvidia/cu13/lib/libcudart.so.13", output)
            self.assertIn('export VLLM_CUDA_RUNTIME_LIB="$vllm_cuda_runtime_lib"', output)
            self.assertIn('export LD_LIBRARY_PATH="$vllm_cuda_runtime_lib:', output)

    def test_ollama_headless_run_pins_version_and_cache_environment(self):
        output = self.make_dry_run(
            "observe-bench", "OBS_RUNTIME=ollama", "OBS_MODEL_SOURCE=local-gguf",
            "OBS_MODEL=/workspace/model.gguf", "OBS_OLLAMA_BIN=/usr/local/bin/ollama",
        )
        self.assertIn('OLLAMA_VERSION="0.34.0"', output)
        self.assertIn('OLLAMA_KEEP_ALIVE="-1"', output)
        self.assertIn('OLLAMA_KV_CACHE_TYPE="f16"', output)
        self.assertIn('OLLAMA_NUM_PARALLEL="1"', output)

    def test_vllm_gguf_plugin_uses_pinned_commit(self):
        output = self.make_dry_run(
            "observe-bench", "OBS_RUNTIME=vllm", "OBS_MODEL_SOURCE=local-gguf",
            "OBS_MODEL=/workspace/model.gguf",
        )
        self.assertIn("e2b8ad532b8b5ea175100202c30430c1d2b5e6a8", output)
        self.assertIn("--no-build-isolation", output)

    def test_model_source_runtime_matrix_rejects_hf_on_llama(self):
        result = subprocess.run(
            ["make", "observe-check-model-source", "OBS_RUNTIME=llama",
             "OBS_MODEL_SOURCE=hf", "OBS_MODEL=owner/model"],
            cwd=ROOT, text=True, capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HF/local-HF só usa vLLM", result.stderr)

    def test_pull_via_ssh_alias_does_not_require_explicit_identity(self):
        output = self.make_dry_run(
            "pull-observe-results", "OBS_REMOTE_HOST=runpod-qwen", "OBS_REMOTE_KEY="
        )
        self.assertIn('if test -n ""; then', output)
        self.assertIn('scp "${ssh_key_args[@]}"', output)


if __name__ == "__main__":
    unittest.main()
