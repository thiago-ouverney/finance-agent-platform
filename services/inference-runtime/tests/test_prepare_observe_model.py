import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_observe_model as subject


class RequestValidationTests(unittest.TestCase):
    def test_hf_sources_are_vllm_only(self):
        for source in ("hf", "local-hf"):
            for runtime in ("llama", "ollama"):
                with self.subTest(source=source, runtime=runtime):
                    with self.assertRaisesRegex(ValueError, "só pode ser usada"):
                        subject.validate_request(
                            source=source,
                            runtime=runtime,
                            model="owner/model",
                            output_dir=Path("models"),
                            gguf_filename=None,
                        )

    def test_gguf_sources_accept_every_runtime(self):
        for source in ("gguf", "local-gguf"):
            for runtime in subject.RUNTIMES:
                with self.subTest(source=source, runtime=runtime):
                    subject.validate_request(
                        source=source,
                        runtime=runtime,
                        model="owner/model" if source == "gguf" else "/models/a.gguf",
                        output_dir=Path("models") if source == "gguf" else None,
                        gguf_filename="a.gguf" if source == "gguf" else None,
                        tokenizer="owner/tokenizer" if source == "gguf" else "/models/tokenizer",
                    )

    def test_remote_source_requires_destination_and_gguf_filename(self):
        with self.assertRaisesRegex(ValueError, "output-dir"):
            subject.validate_request(
                source="hf", runtime="vllm", model="owner/model",
                output_dir=None, gguf_filename=None,
            )
        with self.assertRaisesRegex(ValueError, "gguf-filename"):
            subject.validate_request(
                source="gguf", runtime="llama", model="owner/model",
                output_dir=Path("models"), gguf_filename=None, tokenizer="owner/tokenizer",
            )
        with self.assertRaisesRegex(ValueError, "relativo"):
            subject.validate_request(
                source="gguf", runtime="llama", model="owner/model",
                output_dir=Path("models"), gguf_filename="../secret.gguf",
                tokenizer="owner/tokenizer",
            )
        with self.assertRaisesRegex(ValueError, "tokenizer é obrigatório"):
            subject.validate_request(
                source="local-gguf", runtime="ollama", model="/models/a.gguf",
                output_dir=None, gguf_filename=None,
            )
        with self.assertRaisesRegex(ValueError, "tokenizer-revision exige"):
            subject.validate_request(
                source="local-hf", runtime="vllm", model="/models/hf",
                output_dir=None, gguf_filename=None, tokenizer_revision="main",
            )


class LocalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_local_gguf_validates_signature_and_records_hash(self):
        model = self.root / "model.gguf"
        contents = b"GGUF" + bytes(range(32))
        model.write_bytes(contents)
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
        metadata_path = self.root / "metadata" / "model.json"

        metadata = subject.prepare_model(
            source="local-gguf",
            runtime="ollama",
            model=str(model),
            revision=None,
            gguf_filename=None,
            output_dir=None,
            metadata_out=metadata_path,
            tokenizer=str(tokenizer),
        )

        expected_hash = hashlib.sha256(contents).hexdigest()
        self.assertEqual(metadata["sha256"], expected_hash)
        self.assertEqual(metadata["bytes"], len(contents))
        self.assertEqual(metadata["artifact"]["signature"], "GGUF")
        self.assertEqual(metadata["resolved_revision"], f"sha256:{expected_hash}")
        self.assertEqual(metadata["paths"]["model"], str(model.resolve()))
        self.assertEqual(metadata["local_path"], str(model.resolve()))
        self.assertEqual(metadata["tokenizer_path"], str(tokenizer.resolve()))
        self.assertEqual(metadata["paths"]["tokenizer"], str(tokenizer.resolve()))
        self.assertEqual(metadata["requested_model"], str(model))
        self.assertEqual(json.loads(metadata_path.read_text()), metadata)

    def test_invalid_gguf_does_not_write_metadata(self):
        model = self.root / "broken.gguf"
        model.write_bytes(b"nope")
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
        metadata_path = self.root / "metadata.json"
        with self.assertRaisesRegex(ValueError, "assinatura GGUF inválida"):
            subject.prepare_model(
                source="local-gguf",
                runtime="llama",
                model=str(model),
                revision=None,
                gguf_filename=None,
                output_dir=None,
                metadata_out=metadata_path,
                tokenizer=str(tokenizer),
            )
        self.assertFalse(metadata_path.exists())

    def test_local_hf_hashes_weights_and_captures_tokenizer(self):
        model = self.root / "hf-model"
        model.mkdir()
        (model / "model-00002-of-00002.safetensors").write_bytes(b"second")
        (model / "model-00001-of-00002.safetensors").write_bytes(b"first")
        (model / "training_args.bin").write_bytes(b"not-a-weight")
        (model / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
        (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        metadata = subject.prepare_model(
            source="local-hf",
            runtime="vllm",
            model=str(model),
            revision="experiment-42",
            gguf_filename=None,
            output_dir=self.root / "unused",
            metadata_out=self.root / "identity.json",
        )

        self.assertEqual(metadata["resolved_revision"], "experiment-42")
        self.assertEqual(
            [record["path"] for record in metadata["artifact"]["files"]],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        )
        self.assertEqual(metadata["bytes"], len(b"firstsecond"))
        self.assertTrue(metadata["tokenizer"]["present"])
        self.assertEqual(
            [record["path"] for record in metadata["tokenizer"]["files"]],
            ["tokenizer.json", "tokenizer_config.json"],
        )
        self.assertEqual(metadata["paths"]["tokenizer"], str(model.resolve()))

    def test_hf_directory_without_weights_is_rejected(self):
        model = self.root / "empty-hf"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nenhum arquivo de pesos"):
            subject.prepare_model(
                source="local-hf",
                runtime="vllm",
                model=str(model),
                revision=None,
                gguf_filename=None,
                output_dir=None,
                metadata_out=self.root / "identity.json",
            )


class RemotePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_tokenizer_download_excludes_model_weights(self):
        captured = {}

        def snapshot_download(**kwargs):
            captured.update(kwargs)
            return kwargs["local_dir"]

        fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            destination = self.root / "tokenizer"
            result = subject.download_tokenizer_snapshot(
                "owner/model", "c" * 40, destination
            )

        self.assertEqual(result, destination)
        self.assertIn("tokenizer*", captured["allow_patterns"])
        self.assertIn("*.model", captured["allow_patterns"])
        self.assertNotIn("*.safetensors", captured["allow_patterns"])

    def test_remote_hf_resolves_commit_before_snapshot(self):
        events = []
        commit = "a" * 40

        def resolve(repo_id, revision):
            events.append(("resolve", repo_id, revision))
            return commit

        def download(repo_id, revision, destination):
            events.append(("download", repo_id, revision, destination))
            (destination / "model.safetensors").write_bytes(b"weights")
            (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
            return destination

        old_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "must-never-appear-in-metadata"
        try:
            metadata_path = self.root / "identity.json"
            metadata = subject.prepare_model(
                source="hf",
                runtime="vllm",
                model="owner/model",
                revision=None,
                gguf_filename=None,
                output_dir=self.root / "download",
                metadata_out=metadata_path,
                revision_resolver=resolve,
                snapshot_downloader=download,
            )
        finally:
            if old_token is None:
                os.environ.pop("HF_TOKEN", None)
            else:
                os.environ["HF_TOKEN"] = old_token

        self.assertEqual(events[0], ("resolve", "owner/model", "main"))
        self.assertEqual(events[1][0:3], ("download", "owner/model", commit))
        self.assertEqual(
            events[1][3], (self.root / "download" / "model" / commit).resolve()
        )
        self.assertEqual(metadata["resolved_revision"], commit)
        self.assertIsNone(metadata["requested_revision"])
        self.assertNotIn("must-never-appear-in-metadata", metadata_path.read_text())

    def test_remote_gguf_downloads_pinned_file_then_validates_it(self):
        commit = "b" * 40
        calls = []

        def resolve(repo_id, revision):
            calls.append(("resolve", repo_id, revision))
            return commit

        def download(repo_id, revision, filename, destination):
            calls.append(("download", repo_id, revision, filename))
            path = destination / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"GGUFremote")
            return path

        def download_tokenizer(repo_id, revision, destination):
            calls.append(("download-tokenizer", repo_id, revision))
            (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
            return destination

        metadata = subject.prepare_model(
            source="gguf",
            runtime="llama",
            model="owner/gguf-model",
            revision="release-v1",
            gguf_filename="quant/model.gguf",
            output_dir=self.root / "download",
            metadata_out=self.root / "identity.json",
            tokenizer="owner/base-model",
            tokenizer_revision="tokenizer-v2",
            revision_resolver=resolve,
            file_downloader=download,
            tokenizer_downloader=download_tokenizer,
        )

        self.assertEqual(calls[0], ("resolve", "owner/gguf-model", "release-v1"))
        self.assertEqual(
            calls[1], ("download", "owner/gguf-model", commit, "quant/model.gguf")
        )
        self.assertEqual(calls[2], ("resolve", "owner/base-model", "tokenizer-v2"))
        self.assertEqual(
            calls[3], ("download-tokenizer", "owner/base-model", commit)
        )
        self.assertEqual(metadata["resolved_revision"], commit)
        self.assertEqual(metadata["artifact"]["signature"], "GGUF")
        self.assertEqual(
            metadata["paths"]["model"],
            str(
                (
                    self.root
                    / "download"
                    / "model"
                    / commit
                    / "quant"
                    / "model.gguf"
                ).resolve()
            ),
        )
        self.assertEqual(
            metadata["tokenizer_path"],
            str((self.root / "download" / "tokenizer" / commit).resolve()),
        )


if __name__ == "__main__":
    unittest.main()
