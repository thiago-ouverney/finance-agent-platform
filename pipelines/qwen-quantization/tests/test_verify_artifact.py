from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from verify_artifact import inspect_artifact, reload_artifact, verify_expectations  # noqa: E402


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids="tokens")
        self.device: str | None = None

    def to(self, device: str) -> "FakeInputs":
        self.device = device
        return self


class VerifyArtifactTest(unittest.TestCase):
    def create_artifact(self, root: Path) -> Path:
        weight_bytes = b"weights"
        weight_hash = hashlib.sha256(weight_bytes).hexdigest()
        (root / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {
                        "bits": 4,
                        "group_size": 128,
                        "quant_method": "gptq",
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / "quantization_manifest.json").write_text(
            json.dumps(
                {
                    "quantization_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "source_model_revision": "abc123",
                    "calibration": {"source_sha256": "dataset-hash"},
                    "weight_files": [
                        {
                            "name": "model.safetensors",
                            "size_bytes": len(weight_bytes),
                            "sha256": weight_hash,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(weight_bytes)
        return root

    def test_inspects_complete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = inspect_artifact(self.create_artifact(Path(temporary)))

        self.assertEqual(summary["method"], "gptq")
        self.assertEqual(summary["source_model_revision"], "abc123")
        self.assertEqual(summary["calibration_sha256"], "dataset-hash")

    def test_rejects_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self.create_artifact(Path(temporary))
            (artifact / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Peso registrado ausente"):
                inspect_artifact(artifact)

    def test_rejects_corrupted_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self.create_artifact(Path(temporary))
            (artifact / "model.safetensors").write_bytes(b"weightz")
            with self.assertRaisesRegex(ValueError, "SHA-256 do peso diverge"):
                inspect_artifact(artifact)

    def test_rejects_manifest_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = self.create_artifact(Path(temporary))
            manifest_path = artifact / "quantization_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["bits"] = 8
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bits do manifesto divergem"):
                inspect_artifact(artifact)

    def test_reload_runs_one_token_inference(self) -> None:
        inputs = FakeInputs()
        tokenizer = MagicMock(return_value=inputs)
        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.generate.return_value = [[1, 2]]
        gptq_model = MagicMock()
        gptq_model.from_quantized.return_value = model
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = auto_tokenizer
        gptqmodel = types.ModuleType("gptqmodel")
        gptqmodel.GPTQModel = gptq_model
        gptqmodel.BACKEND = types.SimpleNamespace(GPTQ_TORCH="gptq_torch")

        with patch.dict(sys.modules, {"transformers": transformers, "gptqmodel": gptqmodel}):
            reload_artifact(Path("/artifact"), "gptq", "cuda:0")

        auto_tokenizer.from_pretrained.assert_called_once_with(
            "/artifact", local_files_only=True
        )
        gptq_model.from_quantized.assert_called_once_with(
            "/artifact", device="cuda:0", backend="gptq_torch"
        )
        self.assertEqual(inputs.device, "cuda:0")
        model.generate.assert_called_once_with(
            input_ids="tokens", max_new_tokens=1, do_sample=False
        )

    def test_rejects_make_expectation_mismatch(self) -> None:
        summary = {
            "method": "awq",
            "bits": 4,
            "source_model_id": "Qwen/source",
        }
        with self.assertRaisesRegex(ValueError, "Metodo do artefato diverge"):
            verify_expectations(
                summary,
                method="gptq",
                bits="4",
                source_model_id="Qwen/source",
            )


if __name__ == "__main__":
    unittest.main()
