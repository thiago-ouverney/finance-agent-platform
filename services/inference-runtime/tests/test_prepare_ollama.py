import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_ollama


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"models":[]}')


FAKE_OLLAMA = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
events = Path(os.environ["FAKE_EVENTS"])
with events.open("a") as stream:
    stream.write(json.dumps({"argv": sys.argv[1:], "host": os.environ.get("OLLAMA_HOST")}) + "\n")
if sys.argv[1] == "serve":
    host = os.environ["OLLAMA_HOST"].split("://", 1)[-1]
    address, port = host.rsplit(":", 1)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
    HTTPServer((address, int(port)), Handler).serve_forever()
elif sys.argv[1] == "create":
    text = Path(sys.argv[sys.argv.index("-f") + 1]).read_text()
    with events.open("a") as stream:
        stream.write(json.dumps({"modelfile": text}) + "\n")
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class PrepareOllamaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.gguf = self.root / "model with space.gguf"
        self.gguf.write_bytes(b"GGUF")
        self.binary = self.root / "fake-ollama"
        self.binary.write_text(FAKE_OLLAMA)
        self.binary.chmod(0o755)
        self.events = self.root / "events.jsonl"
        self.old_events = os.environ.get("FAKE_EVENTS")
        os.environ["FAKE_EVENTS"] = str(self.events)

    def tearDown(self):
        if self.old_events is None:
            os.environ.pop("FAKE_EVENTS", None)
        else:
            os.environ["FAKE_EVENTS"] = self.old_events
        self.temp.cleanup()

    def records(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_starts_and_stops_owned_daemon_then_creates_and_shows(self):
        base_url = f"http://127.0.0.1:{free_port()}"
        log = prepare_ollama.prepare(
            str(self.binary), "local-alias", self.gguf, 4096, base_url, 5, self.root / "results"
        )
        records = self.records()
        self.assertEqual([row["argv"][0] for row in records if "argv" in row], ["serve", "create", "show"])
        self.assertTrue(all(row["host"] == base_url for row in records if "argv" in row))
        modelfile = next(row["modelfile"] for row in records if "modelfile" in row)
        self.assertIn(json.dumps(str(self.gguf.resolve())), modelfile)
        self.assertIn("PARAMETER num_ctx 4096", modelfile)
        self.assertTrue(log.is_file())
        self.assertFalse(prepare_ollama.api_ready(base_url, timeout=0.1))

    def test_preserves_preexisting_daemon(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            prepare_ollama.prepare(
                str(self.binary), "alias", self.gguf, 2048, base_url, 2, self.root / "results"
            )
            self.assertEqual([row["argv"][0] for row in self.records() if "argv" in row], ["create", "show"])
            self.assertTrue(prepare_ollama.api_ready(base_url))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_rejects_missing_gguf_before_starting_daemon(self):
        with self.assertRaises(FileNotFoundError):
            prepare_ollama.prepare(
                str(self.binary), "alias", self.root / "missing.gguf", 1024,
                f"http://127.0.0.1:{free_port()}", 1, self.root / "results",
            )
        self.assertFalse(self.events.exists())


if __name__ == "__main__":
    unittest.main()
