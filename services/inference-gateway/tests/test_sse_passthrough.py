import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import uvicorn

from gateway.app import create_app


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SplitStreamUpstream:
    """Envia o primeiro evento SSE e segura o restante."""

    def __init__(self) -> None:
        self.hits = 0
        self.first_sent = threading.Event()
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
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n')
                self.wfile.flush()
                outer.first_sent.set()
                outer.release.wait(timeout=5)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"B"}}]}\n\ndata: [DONE]\n\n')
                self.wfile.flush()

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


class SsePassthroughTests(unittest.TestCase):
    def test_first_token_arrives_before_the_rest_of_the_stream(self):
        upstream = SplitStreamUpstream()
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
        payload = {"model": "mock", "stream": True, "messages": [{"role": "user", "content": "oi"}]}
        received: list[bytes] = []
        done = threading.Event()

        def read() -> None:
            with httpx.Client(timeout=5) as client:
                with client.stream("POST", f"{base}/v1/chat/completions", json=payload) as response:
                    self.assertEqual(response.status_code, 200)
                    for chunk in response.iter_bytes():
                        received.append(chunk)
            done.set()

        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            else:
                self.fail("gateway não subiu")
            threading.Thread(target=read, daemon=True).start()
            self.assertTrue(upstream.first_sent.wait(1))
            self.assertTrue(_saw(received, b"A", timeout=0.5), "o primeiro token ficou preso no gateway")
            self.assertFalse(_saw(received, b"B", timeout=0.05))
            rejected = client.post(f"{base}/v1/chat/completions", json=payload)
            self.assertEqual(rejected.status_code, 429)
            self.assertEqual(upstream.hits, 1)
            upstream.release.set()
            self.assertTrue(done.wait(2))
        self.assertIn(b"B", b"".join(received))


def _saw(chunks: list[bytes], needle: bytes, timeout: float) -> bool:
    deadline = threading.Event()
    timer = threading.Timer(timeout, deadline.set)
    timer.start()
    while not deadline.is_set():
        if needle in b"".join(chunks):
            timer.cancel()
            return True
        threading.Event().wait(0.01)
    return needle in b"".join(chunks)


if __name__ == "__main__":
    unittest.main()
