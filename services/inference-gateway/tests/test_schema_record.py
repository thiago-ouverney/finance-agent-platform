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
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
            "additionalProperties": False,
        },
    },
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FixedUpstream:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self._server = ThreadingHTTPServer(("127.0.0.1", free_port()), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=2)

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(outer.body)))
                self.end_headers()
                self.wfile.write(outer.body)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


class SchemaRecordTests(unittest.TestCase):
    def test_invalid_response_is_recorded_and_returned_unchanged(self):
        raw = b'{"choices":[{"message":{"role":"assistant","content":"nao-e-json"}}]}'
        upstream = FixedUpstream(raw)
        self.addCleanup(upstream.close)
        app = create_app(upstream.base_url, max_inflight=1, max_queue=0, retry_after_s=2)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=2)

        self.addCleanup(stop)
        base = f"http://127.0.0.1:{port}"
        payload = {
            "model": "mock",
            "messages": [{"role": "user", "content": "oi"}],
            "response_format": SCHEMA,
        }
        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            else:
                self.fail("gateway não subiu")
            response = client.post(f"{base}/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, raw)
        self.assertEqual(len(app.state.schema_failures), 1)
        self.assertEqual(app.state.schema_failures[0]["agent_id"], "anonymous")
        self.assertTrue(app.state.schema_failures[0]["reason"])

    def test_matching_schema_records_nothing(self):
        raw = b'{"choices":[{"message":{"role":"assistant","content":"{\\"tags\\":[\\"ok\\"]}"}}]}'
        upstream = FixedUpstream(raw)
        self.addCleanup(upstream.close)
        app = create_app(upstream.base_url, max_inflight=1, max_queue=0, retry_after_s=2)
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=2)

        self.addCleanup(stop)
        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            else:
                self.fail("gateway não subiu")
            client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "mock",
                    "messages": [{"role": "user", "content": "oi"}],
                    "response_format": SCHEMA,
                },
            )
        self.assertEqual(app.state.schema_failures, [])


if __name__ == "__main__":
    unittest.main()
