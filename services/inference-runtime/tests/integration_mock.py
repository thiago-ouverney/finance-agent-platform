"""Teste HTTP/SSE local com GuideLLM real; não usa GPU nem mede um modelo real."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
state = {"active": 0, "maximum": 0, "bodies": [], "mode": "success"}
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            raw = b'vllm:kv_cache_usage_perc{engine="0"} 0.25\nvllm:gpu_cache_usage_perc{engine="0"} 0.25\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        raw = json.dumps({"object": "list", "data": [{"id": "mock-model", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        assert self.path == "/v1/chat/completions", self.path
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] is True
        assert body["temperature"] == 0
        with lock:
            state["bodies"].append(body)
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        if state["mode"] == "failure" and len(state["bodies"]) >= 3:
            raw = b'{"error":{"message":"simulated failure"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            with lock:
                state["active"] -= 1
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for content in ["Teste", " com", " resposta", " curta."]:
                time.sleep(.015)
                event = {"id": "mock", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            event = {"choices": [], "usage": {"prompt_tokens": 256, "completion_tokens": 4, "total_tokens": 260}}
            if state["mode"] == "missing_usage":
                event.pop("usage")
            self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
            self.wfile.flush()
        finally:
            with lock:
                state["active"] -= 1
            self.close_connection = True


def main():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    with tempfile.TemporaryDirectory(prefix="chatbench-test-") as tmp:
        folder = Path(tmp)
        # Artefato de teste, nunca usado como modelo real.
        (folder / "mock.gguf").write_bytes(b"mock-model-fixture")
        vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
        vocab.update({f"word{i}": i+3 for i in range(100)})
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
        tokenizer.save_pretrained(folder / "tokenizer")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        cfg = json.loads((ROOT / "configs/vllm.json").read_text())
        cfg.update(base_url=f"http://127.0.0.1:{server.server_port}", model="mock-model", tokenizer=str(folder / "tokenizer"))
        (folder / "config.json").write_text(json.dumps(cfg))
        try:
            proc = subprocess.run([sys.executable, str(ROOT / "bench.py"), "run", "--config", str(folder / "config.json"),
                                   "--local-model-path", str(folder / "mock.gguf"), "--collect-kv-metrics",
                                   "--smoke", "--scenarios", "short", "--warmup", "3", "--results", str(folder / "results")],
                                  capture_output=True, text=True, timeout=180)
            if proc.returncode:
                print(proc.stdout)
                print(proc.stderr)
                raise AssertionError(f"Exit code {proc.returncode}")
            result = next((folder / "results").glob("**/json/summary.json"))
            rows = json.loads(result.read_text())
            summary = next(row for row in rows if row["phase"] == "measure")
            warmup = next(row for row in rows if row["phase"] == "warmup")
            assert warmup["successful_request_count"] == 3
            assert warmup["requests_sha256"] != summary["requests_sha256"]
            assert summary["successful_request_count"] == 3, summary
            assert summary["output_completion_token_count_p50"] == 4, summary
            assert summary["request_first_token_latency_milliseconds_p50"] > 0, summary
            assert summary["decode_generation_tokens_per_second_p50"] > 0, summary
            assert summary["effective_output_tokens_per_second_p50"] > 0, summary
            assert (result.parent / "context-summary.json").exists()
            assert (result.parent / "gpu-summary.json").exists()
            csv_dir = result.parents[1] / "csv"
            html_dir = result.parents[1] / "html"
            assert (html_dir / "telemetry-memory.svg").exists()
            assert (html_dir / "telemetry-gpu-util.svg").exists()
            kv = (csv_dir / "kv-cache.csv").read_text()
            assert "vllm:kv_cache_usage_perc" in kv, kv
            assert "vllm:gpu_cache_usage_perc" not in kv, kv
            assert state["maximum"] == 1, state
            assert len(state["bodies"]) == 8, len(state["bodies"])
            assert state["bodies"][0] == state["bodies"][-1]
            measured_prompts = [body["messages"][0]["content"] for body in state["bodies"][1:-1]]
            assert len(set(measured_prompts)) == 6, "warmup e medidas devem usar prompts distintos"
            lifecycle = json.loads((result.parent / "lifecycle.json").read_text())
            assert lifecycle["first_request"]["ttft_ms"] > 0
            assert lifecycle["warm_reference"]["ttft_ms"] > 0
            assert lifecycle["readiness"]["process_to_api_observed_s"] is None
            assert lifecycle["mode"] == "existing-server-state-unknown"
            assert json.loads((result.parent / "manifest.json").read_text())["status"] == "complete"
            print("PASS: primeira resposta + 3 warmups + 3 medidas distintas + referência final; concorrência máxima = 1.", flush=True)
            assert (csv_dir / "r1-short-measure-requests.csv").exists()
            for mode in ("failure", "missing_usage"):
                state.update(mode=mode, bodies=[])
                proc = subprocess.run([sys.executable, str(ROOT / "bench.py"), "run", "--config", str(folder / "config.json"),
                                       "--local-model-path", str(folder / "mock.gguf"),
                                       "--smoke", "--scenarios", "short", "--warmup", "1", "--results", str(folder / mode)],
                                      capture_output=True, text=True, timeout=180)
                if mode == "missing_usage":
                    assert proc.returncode == 0, (mode, proc.stdout, proc.stderr)
                    result = next((folder / mode).glob("**/json/summary.json"))
                    rows = json.loads(result.read_text())
                    measure = next(row for row in rows if row["phase"] == "measure")
                    assert measure["decode_tokens_per_second_sample_count"] == 0, measure
                    assert measure["end_to_end_tokens_per_second_p50"] is None, measure
                else:
                    assert proc.returncode != 0, (mode, proc.stdout)
                manifest = next((folder / mode).glob("**/json/manifest.json"))
                expected_status = "complete" if mode == "missing_usage" else "failed"
                assert json.loads(manifest.read_text())["status"] == expected_status
                if mode == "failure":
                    failed_summary = next(row for row in json.loads((manifest.parent / "summary.json").read_text()) if row["phase"] == "measure")
                    assert failed_summary["errored_request_count"] >= 1, failed_summary
                else:
                    missing_summary = json.loads((manifest.parent / "summary.json").read_text())
                    assert missing_summary, missing_summary
                    partial = json.loads((manifest.parent / "first-request.json").read_text())
                    assert partial["status"] == "complete"
                    assert partial["ttft_ms"] > 0
                    assert partial["usage_observed"] is False
                print(f"PASS: {mode} tratado explicitamente sem inventar métricas.")
            with socket.socket() as unused:
                unused.bind(("127.0.0.1", 0))
                port = unused.getsockname()[1]
            cfg.update(base_url=f"http://127.0.0.1:{port}", model="mock")
            (folder / "config.json").write_text(json.dumps(cfg))
            (folder / "launch.json").write_text(json.dumps([sys.executable, str(ROOT / "tests/test_lifecycle.py"), "--serve", str(port)]))
            proc = subprocess.run([sys.executable, str(ROOT / "bench.py"), "run", "--config", str(folder / "config.json"),
                                   "--local-model-path", str(folder / "mock.gguf"),
                                   "--launch", str(folder / "launch.json"), "--smoke", "--input-tokens", "256", "--warmup", "1",
                                   "--results", str(folder / "startup")], capture_output=True, text=True, timeout=180)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            lifecycle_path = next((folder / "startup").glob("**/json/lifecycle.json"))
            lifecycle = json.loads(lifecycle_path.read_text())
            assert lifecycle["status"] == "complete"
            assert lifecycle["mode"] == "new-process"
            assert lifecycle["first_request"]["process_to_first_content_s"] > lifecycle["readiness"]["process_to_api_observed_s"]
            assert "server_cleanup" in lifecycle
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0, "O servidor de teste deveria estar encerrado."
            print("PASS: CLI --launch completo, cronômetro desde a partida, GuideLLM e limpeza do servidor criado.")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
