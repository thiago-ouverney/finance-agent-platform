import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from gateway.app import create_app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class HoldingUpstream:
    """Aceita o POST e só responde quando o teste libera."""

    def __init__(self) -> None:
        self.hits = 0
        self._lock = threading.Lock()
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
                with outer._lock:
                    outer.hits += 1
                outer.release.wait(timeout=5)
                body = json.dumps(
                    {"id": "mock", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


class GatewayHttpTests(unittest.TestCase):
    def test_full_queue_returns_429_and_skips_the_upstream(self):
        upstream = HoldingUpstream()
        self.addCleanup(upstream.close)
        app = create_app(upstream.base_url, max_inflight=1, max_queue=1, retry_after_s=2)
        import uvicorn

        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=2)

        self.addCleanup(stop)
        base = f"http://127.0.0.1:{port}"
        payload = {"model": "mock", "messages": [{"role": "user", "content": "oi"}]}
        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            else:
                self.fail("gateway não subiu")

            first: dict = {}
            second: dict = {}

            def call(slot: dict) -> None:
                response = client.post(f"{base}/v1/chat/completions", json=payload)
                slot["status"] = response.status_code

            inflight = threading.Thread(target=call, args=(first,))
            queued = threading.Thread(target=call, args=(second,))
            inflight.start()
            for _ in range(50):
                if upstream.hits == 1:
                    break
                threading.Event().wait(0.02)
            self.assertEqual(upstream.hits, 1)
            queued.start()
            threading.Event().wait(0.1)
            rejected = client.post(f"{base}/v1/chat/completions", json=payload)
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(rejected.headers["retry-after"], "2")
            self.assertEqual(upstream.hits, 1)
            upstream.release.set()
            inflight.join(timeout=2)
            queued.join(timeout=2)
        self.assertEqual(first["status"], 200)
        self.assertEqual(second["status"], 200)
        self.assertEqual(upstream.hits, 2)


if __name__ == "__main__":
    unittest.main()
