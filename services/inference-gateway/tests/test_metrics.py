import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import uvicorn

from gateway.app import create_app

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tags",
        "schema": {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
            "required": ["tags"],
        },
    },
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def samples(text: str) -> dict[str, float]:
    found = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        found[name] = float(value)
    return found


class HoldingUpstream:
    def __init__(self) -> None:
        self.hits = 0
        self.release = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", free_port()), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.release.set()
        self._server.shutdown()
        self._thread.join(timeout=2)

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                outer.hits += 1
                outer.release.wait(timeout=5)
                body = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": "nao-e-json"}}]}
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


class MetricsTests(unittest.TestCase):
    def test_metrics_expose_inflight_queue_rejections_and_schema_failures(self):
        upstream = HoldingUpstream()
        self.addCleanup(upstream.close)
        app = create_app(
            upstream.base_url,
            max_inflight=1,
            max_queue=1,
            retry_after_s=2,
            queue_timeout_s=0.4,
        )
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=2)

        self.addCleanup(stop)
        base = f"http://127.0.0.1:{port}"
        schema_payload = {
            "model": "mock",
            "messages": [{"role": "user", "content": "oi"}],
            "response_format": SCHEMA,
        }
        plain = {"model": "mock", "messages": [{"role": "user", "content": "fila"}]}

        def post(payload: dict, slot: dict, key: str) -> None:
            with httpx.Client(timeout=5) as client:
                slot[key] = client.post(f"{base}/v1/chat/completions", json=payload).status_code

        held: dict = {}
        queued: dict = {}
        threading.Thread(target=post, args=(schema_payload, held, "status"), daemon=True).start()
        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"{base}/health").status_code == 200 and upstream.hits == 1:
                        break
                except httpx.HTTPError:
                    pass
                threading.Event().wait(0.02)
            else:
                self.fail("pedido em voo não chegou ao upstream")
            threading.Thread(target=post, args=(plain, queued, "status"), daemon=True).start()
            seen_queue = False
            for _ in range(20):
                metrics = samples(client.get(f"{base}/metrics").text)
                if metrics.get("gateway_queue") == 1 and metrics.get("gateway_inflight") == 1:
                    seen_queue = True
                    break
                threading.Event().wait(0.02)
            self.assertTrue(seen_queue)
            rejected = client.post(f"{base}/v1/chat/completions", json=plain)
            self.assertEqual(rejected.status_code, 429)
            during = samples(client.get(f"{base}/metrics").text)
            self.assertEqual(during['gateway_rejections_total{status="429"}'], 1)
            for _ in range(30):
                if queued.get("status") == 503:
                    break
                threading.Event().wait(0.02)
            self.assertEqual(queued.get("status"), 503)
            upstream.release.set()
            for _ in range(50):
                if held.get("status") == 200:
                    break
                threading.Event().wait(0.02)
            final = samples(client.get(f"{base}/metrics").text)
        self.assertEqual(final["gateway_inflight"], 0)
        self.assertEqual(final["gateway_queue"], 0)
        self.assertEqual(final['gateway_rejections_total{status="429"}'], 1)
        self.assertEqual(final['gateway_rejections_total{status="503"}'], 1)
        self.assertEqual(final["gateway_schema_failures_total"], 1)


if __name__ == "__main__":
    unittest.main()
