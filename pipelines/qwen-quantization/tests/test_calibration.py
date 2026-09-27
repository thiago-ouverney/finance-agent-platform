from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from calibration import (  # noqa: E402
    GoldenExample,
    inspect_dataset,
    load_golden_csv,
    messages_for_example,
    render_calibration_texts,
    select_calibration_examples,
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
        self.assertions = (tokenize, add_generation_prompt)
        return "\n".join(f"{message['role']}:{message['content']}" for message in conversation)


class GoldenCalibrationTest(unittest.TestCase):
    def write_dataset(self, root: Path, rows: list[dict[str, str]], name: str = "train.csv") -> Path:
        path = root / name
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["id", "business_model", "tags"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def valid_rows(self) -> list[dict[str, str]]:
        return [
            {
                "id": "exemplo-a",
                "business_model": "Empresa de tecnologia com atuação nacional.",
                "tags": json.dumps(["tecnologia", "atuação nacional"], ensure_ascii=False),
            },
            {
                "id": "exemplo-b",
                "business_model": "Consultoria B2B prestada remotamente.",
                "tags": json.dumps(["b2b", "serviço", "vendas remotas"], ensure_ascii=False),
            },
            {
                "id": "exemplo-c",
                "business_model": "Registro deliberadamente sem rótulos.",
                "tags": "[]",
            },
        ]

    def test_loads_utf8_golden_and_preserves_empty_tag_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_dataset(Path(temporary), self.valid_rows())
            examples = load_golden_csv(path)

        self.assertEqual(len(examples), 3)
        self.assertEqual(examples[0].tags, ("tecnologia", "atuação nacional"))
        self.assertEqual(examples[2].tags, ())

    def test_selection_is_stable_across_input_order(self) -> None:
        examples = [
            GoldenExample(str(index), f"modelo {index}", ("tag",))
            for index in range(10)
        ]
        first = select_calibration_examples(examples, limit=4, seed=42)
        second = select_calibration_examples(list(reversed(examples)), limit=4, seed=42)
        other_seed = select_calibration_examples(examples, limit=4, seed=43)

        self.assertEqual([item.example_id for item in first], [item.example_id for item in second])
        self.assertNotEqual([item.example_id for item in first], [item.example_id for item in other_seed])

    def test_renders_user_and_expected_assistant_with_chat_template(self) -> None:
        example = GoldenExample("one", "Negócio com café e software.", ("produto", "tecnologia"))
        tokenizer = FakeTokenizer()

        rendered = render_calibration_texts([example], tokenizer)
        messages = messages_for_example(example)

        self.assertIn("Negócio com café", rendered[0])
        self.assertIn('["produto", "tecnologia"]', rendered[0])
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(tokenizer.assertions, (False, False))

    def test_rejects_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_dataset(Path(temporary), self.valid_rows(), name="test.csv")
            with self.assertRaisesRegex(ValueError, "reservado para avaliacao final"):
                load_golden_csv(path)

    def test_rejects_duplicate_ids_and_invalid_tags(self) -> None:
        duplicate = self.valid_rows()[:2]
        duplicate[1] = {**duplicate[1], "id": duplicate[0]["id"]}
        invalid = self.valid_rows()[:1]
        invalid[0] = {**invalid[0], "tags": '{"tag": "b2b"}'}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "id duplicado"):
                load_golden_csv(self.write_dataset(root, duplicate, name="duplicate.csv"))
            with self.assertRaisesRegex(ValueError, "lista de strings"):
                load_golden_csv(self.write_dataset(root, invalid, name="invalid.csv"))

    def test_accepts_example_id_compatibility_column(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.csv"
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=["example_id", "business_model", "tags"])
                writer.writeheader()
                writer.writerow({"example_id": "legacy", "business_model": "B2B", "tags": '["b2b"]'})

            examples = load_golden_csv(path)

        self.assertEqual(examples[0].example_id, "legacy")

    def test_inspection_requires_requested_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_dataset(Path(temporary), self.valid_rows())
            with self.assertRaisesRegex(ValueError, "menos que os 4 solicitados"):
                inspect_dataset(path, limit=4, seed=42)

    def test_rejects_tag_outside_taxonomy(self) -> None:
        rows = self.valid_rows()[:1]
        rows[0] = {**rows[0], "tags": '["tag inventada"]'}
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_dataset(Path(temporary), rows)
            with self.assertRaisesRegex(ValueError, "fora da taxonomia"):
                load_golden_csv(path)


if __name__ == "__main__":
    unittest.main()
