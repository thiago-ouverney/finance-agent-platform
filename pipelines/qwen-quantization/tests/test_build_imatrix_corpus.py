from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from build_imatrix_corpus import (  # noqa: E402
    build_imatrix_corpus,
    load_local_tokenizer,
    serialize_corpus,
    tokenizer_sha256,
    verify_imatrix_corpus,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.conversations.append(conversation)
        if tokenize or add_generation_prompt:
            raise AssertionError("O corpus deve ser texto completo, sem novo prompt de geracao")
        return "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in conversation
        )


class ImatrixCorpusTest(unittest.TestCase):
    def write_dataset(
        self,
        root: Path,
        rows: list[dict[str, str]],
        name: str = "train.csv",
    ) -> Path:
        path = root / name
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["id", "business_model", "tags"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_tokenizer(self, root: Path) -> Path:
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text('{"version":"fake"}\n', encoding="utf-8")
        (tokenizer / "tokenizer_config.json").write_text(
            '{"chat_template":"fake"}\n', encoding="utf-8"
        )
        # A tokenizer hash must not scan multi-GB model weights or local caches.
        (tokenizer / "model.safetensors").write_bytes(b"not-tokenizer-input")
        (tokenizer / ".cache").mkdir()
        (tokenizer / ".cache" / "mutable").write_text("ignored", encoding="utf-8")
        return tokenizer

    def rows(self) -> list[dict[str, str]]:
        return [
            {
                "id": "empresa-a",
                "business_model": "Consultoria B2B com atuação nacional.",
                "tags": json.dumps(["b2b", "serviço", "atuação nacional"], ensure_ascii=False),
            },
            {
                "id": "empresa-b",
                "business_model": "Loja física vende produtos ao consumidor final.",
                "tags": json.dumps(["estabelecimento físico", "produto", "b2c"], ensure_ascii=False),
            },
            {
                "id": "empresa-c",
                "business_model": "Software vendido remotamente para pequenas empresas.",
                "tags": json.dumps(
                    ["tecnologia", "produto", "b2b", "vendas remotas"],
                    ensure_ascii=False,
                ),
            },
        ]

    def build(self, root: Path, *, rows=None, name="train.csv", limit=2, seed=42):
        dataset = self.write_dataset(root, rows or self.rows(), name=name)
        tokenizer_dir = self.write_tokenizer(root)
        corpus = root / "artifacts" / "imatrix-corpus.txt"
        manifest = root / "artifacts" / "imatrix-corpus.manifest.json"
        fake = FakeTokenizer()
        metadata = build_imatrix_corpus(
            dataset,
            tokenizer_dir,
            corpus,
            manifest,
            limit=limit,
            seed=seed,
            tokenizer=fake,
        )
        return dataset, tokenizer_dir, corpus, manifest, fake, metadata

    def test_builds_utf8_chat_corpus_and_private_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _, corpus, manifest, fake, metadata = self.build(root)

            payload = corpus.read_bytes()
            text = payload.decode("utf-8")
            stored = json.loads(manifest.read_text(encoding="utf-8"))
            dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()

        self.assertEqual(stored, metadata)
        self.assertEqual(metadata["dataset_sha256"], dataset_sha256)
        self.assertEqual(metadata["corpus_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata["available_examples"], 3)
        self.assertEqual(metadata["selected_examples"], 2)
        self.assertEqual(metadata["selection_seed"], 42)
        self.assertTrue(metadata["includes_expected_answer"])
        self.assertIn("<|im_start|>user", text)
        self.assertIn("<|im_start|>assistant", text)
        self.assertIn("<|im_end|>", text)
        self.assertIn("atuação", text)
        self.assertEqual(len(fake.conversations), 2)
        self.assertTrue(
            all(
                [message["role"] for message in item] == ["user", "assistant"]
                for item in fake.conversations
            )
        )
        serialized_manifest = json.dumps(stored, ensure_ascii=False)
        for row in self.rows():
            self.assertNotIn(row["id"], serialized_manifest)
            self.assertNotIn(row["business_model"], serialized_manifest)
        self.assertNotIn("tags", stored)

    def test_selection_and_corpus_are_stable_across_input_order(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_temp,
            tempfile.TemporaryDirectory() as second_temp,
        ):
            first = self.build(Path(first_temp), rows=self.rows(), limit=2, seed=7)
            second = self.build(Path(second_temp), rows=list(reversed(self.rows())), limit=2, seed=7)

            first_corpus = first[2].read_bytes()
            second_corpus = second[2].read_bytes()
            first_metadata = first[5]
            second_metadata = second[5]

        self.assertEqual(first_corpus, second_corpus)
        self.assertEqual(first_metadata["selection_sha256"], second_metadata["selection_sha256"])
        self.assertEqual(first_metadata["corpus_sha256"], second_metadata["corpus_sha256"])
        self.assertNotEqual(first_metadata["dataset_sha256"], second_metadata["dataset_sha256"])

    def test_rejects_test_split_and_insufficient_examples_without_outputs(self) -> None:
        for name, limit, expected in (
            ("test.csv", 2, "reservado para avaliacao final"),
            ("train.csv", 4, "menos que os 4 solicitados"),
        ):
            with (
                self.subTest(name=name, limit=limit),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                dataset = self.write_dataset(root, self.rows(), name=name)
                tokenizer_dir = self.write_tokenizer(root)
                corpus = root / "corpus.txt"
                manifest = root / "manifest.json"
                with self.assertRaisesRegex(ValueError, expected):
                    build_imatrix_corpus(
                        dataset,
                        tokenizer_dir,
                        corpus,
                        manifest,
                        limit=limit,
                        seed=42,
                        tokenizer=FakeTokenizer(),
                    )
                self.assertFalse(corpus.exists())
                self.assertFalse(manifest.exists())

    def test_refuses_to_overwrite_either_output(self) -> None:
        for occupied in ("corpus", "manifest"):
            with (
                self.subTest(occupied=occupied),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                dataset = self.write_dataset(root, self.rows())
                tokenizer_dir = self.write_tokenizer(root)
                corpus = root / "corpus.txt"
                manifest = root / "manifest.json"
                target = corpus if occupied == "corpus" else manifest
                target.write_text("preserve-me", encoding="utf-8")

                with self.assertRaisesRegex(FileExistsError, "nao sera sobrescrita"):
                    build_imatrix_corpus(
                        dataset,
                        tokenizer_dir,
                        corpus,
                        manifest,
                        limit=2,
                        seed=42,
                        tokenizer=FakeTokenizer(),
                    )

                self.assertEqual(target.read_text(encoding="utf-8"), "preserve-me")
                other = manifest if occupied == "corpus" else corpus
                self.assertFalse(other.exists())

    def test_verifies_existing_corpus_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, tokenizer_dir, corpus, manifest, _, metadata = self.build(root)

            verified = verify_imatrix_corpus(
                dataset,
                tokenizer_dir,
                corpus,
                manifest,
                limit=2,
                seed=42,
                tokenizer=FakeTokenizer(),
            )
            self.assertEqual(verified, metadata)

            corpus.write_text("alterado\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Corpus imatrix diverge"):
                verify_imatrix_corpus(
                    dataset,
                    tokenizer_dir,
                    corpus,
                    manifest,
                    limit=2,
                    seed=42,
                    tokenizer=FakeTokenizer(),
                )

    def test_tokenizer_hash_uses_only_local_tokenizer_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_dir = self.write_tokenizer(root)
            before, count = tokenizer_sha256(tokenizer_dir)
            (tokenizer_dir / "model.safetensors").write_bytes(b"changed-weight")
            (tokenizer_dir / ".cache" / "mutable").write_text("changed-cache", encoding="utf-8")
            after_ignored_changes, after_count = tokenizer_sha256(tokenizer_dir)
            (tokenizer_dir / "tokenizer.json").write_text('{"version":"changed"}\n', encoding="utf-8")
            after_tokenizer_change, _ = tokenizer_sha256(tokenizer_dir)

        self.assertEqual(count, 2)
        self.assertEqual(after_count, 2)
        self.assertEqual(before, after_ignored_changes)
        self.assertNotEqual(before, after_tokenizer_change)

    def test_local_tokenizer_loader_forbids_network_fallback(self) -> None:
        calls = []

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((path, kwargs))
                return "loaded"

        fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            sys.modules,
            {"transformers": fake_transformers},
        ):
            loaded = load_local_tokenizer(Path(temporary))

        self.assertEqual(loaded, "loaded")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["local_files_only"], True)
        self.assertEqual(calls[0][1]["trust_remote_code"], False)

    def test_serialization_preserves_special_tokens_and_is_deterministic(self) -> None:
        payload = serialize_corpus(
            [
                "<|im_start|>user\nOlá<|im_end|>\n",
                "<|im_start|>assistant\n['serviço']<|im_end|>\n",
            ]
        )
        self.assertEqual(
            payload.decode("utf-8"),
            "<|im_start|>user\nOlá<|im_end|>\n\n"
            "<|im_start|>assistant\n['serviço']<|im_end|>\n",
        )


if __name__ == "__main__":
    unittest.main()
