import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bench
import render_observe_runtime as subject


class RenderRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = self.root / "tokenizer with space"
        self.tokenizer.mkdir()
        (self.tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (self.tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def metadata(self, *, runtime: str, source: str) -> Path:
        if source in subject.HF_SOURCES:
            model = self.root / "model snapshot"
            model.mkdir(exist_ok=True)
            payload = b"hf-weights"
            (model / "model.safetensors").write_bytes(payload)
        else:
            model = self.root / "model with space.gguf"
            payload = b"GGUF" + b"weights"
            model.write_bytes(payload)
        metadata = {
            "schema_version": 1,
            "source": source,
            "runtime": runtime,
            "requested_model": "owner/model",
            "requested_revision": "release-v1",
            "resolved_revision": "a" * 40,
            "local_path": str(model.resolve()),
            "tokenizer_path": str(self.tokenizer.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            # Campos arbitrários/secretos nunca são copiados para as saídas.
            "api_key": "must-not-leak",
            "authorization": "Bearer must-not-leak",
        }
        path = self.root / f"metadata-{runtime}-{source}.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return path

    def render(self, *, runtime: str, source: str, port: int, **overrides):
        config_path = self.root / f"out/{runtime}/config.json"
        launch_path = self.root / f"out/{runtime}/launch.json"
        arguments = {
            "metadata_path": self.metadata(runtime=runtime, source=source),
            "config_out": config_path,
            "launch_out": launch_path,
            "runtime": runtime,
            "model_alias": "bench-model",
            "context": 8192,
            "host": "127.0.0.1",
            "port": port,
            "runtime_bin": f"/opt/{runtime} runtime/bin/server",
            "runtime_version": f"{runtime} test-version",
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 1,
            "dtype": "auto",
            "gpu_layers": 123,
        }
        arguments.update(overrides)
        config, launch = subject.render_runtime(**arguments)
        return config, launch, config_path, launch_path

    def test_vllm_hf_writes_exact_bench_contract_and_local_argv(self):
        config, launch, config_path, launch_path = self.render(
            runtime="vllm", source="hf", port=8000,
            runtime_version="vLLM 0.11.2",
        )

        self.assertEqual(bench.load_config(config_path), config)
        self.assertEqual(json.loads(launch_path.read_text()), launch)
        self.assertEqual(
            launch,
            [
                "/opt/vllm runtime/bin/server",
                "serve",
                str((self.root / "model snapshot").resolve()),
                "--tokenizer",
                str(self.tokenizer.resolve()),
                "--served-model-name",
                "bench-model",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--max-model-len",
                "8192",
                "--max-num-seqs",
                "1",
                "--gpu-memory-utilization",
                "0.85",
                "--dtype",
                "auto",
            ],
        )
        self.assertNotIn("--load-format", launch)
        self.assertEqual(shlex.split(config["server_command"]), launch)
        self.assertEqual(config["base_url"], "http://127.0.0.1:8000")
        self.assertEqual(config["runtime_version"], "vLLM 0.11.2")
        self.assertIn("resolved_revision=" + "a" * 40, config["model_artifact"])
        self.assertIn("sha256=", config["model_artifact"])
        self.assertNotIn("must-not-leak", config_path.read_text())
        self.assertNotIn("must-not-leak", launch_path.read_text())

    def test_vllm_gguf_adds_explicit_format_and_quantization(self):
        config, launch, _config_path, _launch_path = self.render(
            runtime="vllm", source="gguf", port=8000
        )
        model_index = launch.index(str((self.root / "model with space.gguf").resolve()))
        self.assertEqual(
            launch[model_index + 1:model_index + 7],
            [
                "--load-format",
                "gguf",
                "--quantization",
                "gguf",
                "--tokenizer",
                str(self.tokenizer.resolve()),
            ],
        )
        self.assertIn("GGUF no vLLM é experimental", config["notes"])

    def test_llama_uses_single_parallel_slot_and_gpu_layers(self):
        config, launch, config_path, _launch_path = self.render(
            runtime="llama", source="local-gguf", port=8080
        )
        self.assertEqual(
            launch,
            [
                "/opt/llama runtime/bin/server",
                "-m",
                str((self.root / "model with space.gguf").resolve()),
                "--alias",
                "bench-model",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--ctx-size",
                "8192",
                "--n-gpu-layers",
                "123",
                "--parallel",
                "1",
                "--jinja",
            ],
        )
        self.assertEqual(config["runtime"], "llama.cpp")
        self.assertEqual(bench.load_config(config_path), config)

    def test_ollama_launch_is_only_serve_and_alias_is_in_config(self):
        config, launch, config_path, _launch_path = self.render(
            runtime="ollama",
            source="gguf",
            port=11434,
            runtime_bin="/opt/ollama",
        )
        self.assertEqual(launch, ["/opt/ollama", "serve"])
        self.assertEqual(config["model"], "bench-model")
        self.assertIn("Alias bench-model", config["notes"])
        self.assertEqual(bench.load_config(config_path), config)

    def test_rejects_runtime_mismatch_and_hf_on_llama(self):
        metadata_path = self.metadata(runtime="vllm", source="hf")
        metadata = json.loads(metadata_path.read_text())
        with self.assertRaisesRegex(ValueError, "diverge"):
            subject.validate_metadata(metadata, "llama")

        metadata["runtime"] = "llama"
        with self.assertRaisesRegex(ValueError, "exige source GGUF"):
            subject.validate_metadata(metadata, "llama")

    def test_rejects_external_host_and_nondefault_ollama_endpoint(self):
        identity = subject.validate_metadata(
            json.loads(self.metadata(runtime="vllm", source="hf").read_text()),
            "vllm",
        )
        with self.assertRaisesRegex(ValueError, "loopback"):
            subject.build_launch(
                identity=identity,
                runtime="vllm",
                model_alias="model",
                context=4096,
                host="0.0.0.0",
                port=8000,
                runtime_bin="vllm",
                gpu_memory_utilization=0.9,
                max_num_seqs=1,
                dtype="auto",
                gpu_layers=999,
            )

        identity = subject.validate_metadata(
            json.loads(self.metadata(runtime="ollama", source="gguf").read_text()),
            "ollama",
        )
        with self.assertRaisesRegex(ValueError, "endpoint padrão"):
            subject.build_launch(
                identity=identity,
                runtime="ollama",
                model_alias="model",
                context=4096,
                host="127.0.0.1",
                port=12000,
                runtime_bin="ollama",
                gpu_memory_utilization=0.9,
                max_num_seqs=1,
                dtype="auto",
                gpu_layers=999,
            )

    def test_invalid_metadata_preserves_existing_outputs(self):
        metadata_path = self.metadata(runtime="vllm", source="hf")
        metadata = json.loads(metadata_path.read_text())
        metadata["sha256"] = "not-a-hash"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        config_path = self.root / "config.json"
        launch_path = self.root / "launch.json"
        config_path.write_text("old-config", encoding="utf-8")
        launch_path.write_text("old-launch", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "64 dígitos"):
            subject.render_runtime(
                metadata_path=metadata_path,
                config_out=config_path,
                launch_out=launch_path,
                runtime="vllm",
                model_alias="model",
                context=4096,
                host="127.0.0.1",
                port=8000,
                runtime_bin="vllm",
            )

        self.assertEqual(config_path.read_text(), "old-config")
        self.assertEqual(launch_path.read_text(), "old-launch")


if __name__ == "__main__":
    unittest.main()
