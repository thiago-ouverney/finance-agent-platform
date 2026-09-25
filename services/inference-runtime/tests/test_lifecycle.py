import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lifecycle import Launch, timed_request, wait_models


class LifecycleTests(unittest.TestCase):
    def test_wrong_alias_fails_immediately(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "model.gguf"}]}
        with patch("lifecycle.httpx.Client") as client:
            client.return_value.__enter__.return_value.get.return_value = response
            with self.assertRaisesRegex(RuntimeError, "alias"):
                wait_models({"base_url": "http://localhost:8080", "model": "qwen7b"}, "", 1800)

    def test_ollama_latest_tag_is_explicitly_supported(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "qwen7b:latest"}]}
        with patch("lifecycle.httpx.Client") as client:
            client.return_value.__enter__.return_value.get.return_value = response
            result = wait_models({"base_url": "http://localhost:11434", "model": "qwen7b", "runtime": "ollama"}, "", 1)
            self.assertEqual(result["models"], ["qwen7b:latest"])

    def test_refuses_occupied_port_without_killing_anything(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as tmp:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            path = Path(tmp) / "argv.json"
            path.write_text(json.dumps([sys.executable, "-c", "pass"]))
            launch = Launch({"base_url": f"http://127.0.0.1:{listener.getsockname()[1]}"}, path, tmp)
            with self.assertRaisesRegex(ValueError, "porta"):
                launch.start()
            self.assertIsNone(launch.process)
            launch.close()

    def test_spawn_exit_detected_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
            free.close()
            path = Path(tmp) / "argv.json"
            path.write_text(json.dumps([sys.executable, "-c", "raise SystemExit(7)"]))
            cfg = {"base_url": f"http://127.0.0.1:{port}", "model": "mock"}
            launch = Launch(cfg, path, tmp)
            launch.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "encerrou"):
                    wait_models(cfg, "", 3, launch)
            finally:
                launch.close()
            self.assertIsNotNone(launch.process.poll())

    def test_started_process_readiness_and_first_content(self):
        # O subprocesso executa este mesmo arquivo como um pequeno servidor de teste.
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
            free.close()
            path = Path(tmp) / "argv.json"
            path.write_text(json.dumps([sys.executable, str(Path(__file__).resolve()), "--serve", str(port)]))
            cfg = {"base_url": f"http://127.0.0.1:{port}", "model": "mock"}
            launch = Launch(cfg, path, tmp)
            origin = launch.start()
            try:
                ready = wait_models(cfg, "", 10, launch)
                result = timed_request(cfg, "", 3, "Pergunta", Path(tmp) / "first.json", origin)
                self.assertGreater(ready["process_to_api_observed_s"], 0)
                self.assertGreater(result["process_to_first_content_s"], ready["process_to_api_observed_s"])
                self.assertGreaterEqual(result["process_to_response_end_s"], result["process_to_first_content_s"])
                self.assertEqual(result["status"], "complete")
            finally:
                launch.close()
            self.assertIsNotNone(launch.process.poll())


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        raw = b'{"data":[{"id":"mock"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        import time
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # role vazio não deve disparar TTFT.
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.wfile.flush()
        time.sleep(.05)
        event = {"choices": [{"delta": {"content": "Resposta"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
        self.wfile.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
        self.wfile.flush()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), MockHandler).serve_forever()
    else:
        unittest.main()
