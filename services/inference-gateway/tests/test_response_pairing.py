import json
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


class EchoUpstream:
    """Devolve o texto do pedido, para mostrar de quem é a resposta."""

    def __init__(self) -> None:
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
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                incoming = json.loads(self.rfile.read(length))
                content = incoming["messages"][0]["content"]
                body = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": content}}]}
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


def _answers(base: str) -> dict[str, str]:
    found: dict[str, str] = {}

    def ask(agent_id: str) -> None:
        payload = {"model": "mock", "messages": [{"role": "user", "content": agent_id}]}
        with httpx.Client(timeout=5) as client:
            response = client.post(
                f"{base}/v1/chat/completions",
                json=payload,
                headers={"X-Agent-Id": agent_id},
            )
        found[agent_id] = response.json()["choices"][0]["message"]["content"]

    threads = [threading.Thread(target=ask, args=(agent,)) for agent in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
    return found


class ResponsePairingTests(unittest.TestCase):
    def _serve(self, swap_responses: bool) -> tuple[str, object]:
        upstream = EchoUpstream()
        self.addCleanup(upstream.close)
        app = create_app(
            upstream.base_url,
            max_inflight=2,
            max_queue=0,
            retry_after_s=2,
            swap_responses=swap_responses,
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
        with httpx.Client(timeout=5) as client:
            for _ in range(50):
                try:
                    if client.get(f"{base}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
            else:
                self.fail("gateway não subiu")
        return base, server

    def test_normal_path_keeps_each_response_with_its_agent(self):
        base, _server = self._serve(False)
        self.assertEqual(_answers(base), {"A": "A", "B": "B"})

    def test_injected_swap_crosses_the_two_responses(self):
        base, _server = self._serve(True)
        self.assertEqual(_answers(base), {"A": "B", "B": "A"})


if __name__ == "__main__":
    unittest.main()
