import json
import tempfile
import unittest
from pathlib import Path

from analytics.src.build_mopep_runtime_workload import build_workload


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"].split()


class BuildMopepRuntimeWorkloadTests(unittest.TestCase):
    def write_dataset(self, path: Path, prefix: str, sizes: list[int]) -> None:
        rows = ["example_id,business_model,tags"]
        for index, size in enumerate(sizes):
            rows.append(f'{prefix}-{index},"{"palavra " * size}","[\'b2b\']"')
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_builds_deterministic_private_workload_and_frozen_buckets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.csv"
            train = root / "train.csv"
            test = root / "test.csv"
            self.write_dataset(calibration, "cal", [1, 5, 10, 20, 40, 80])
            self.write_dataset(train, "warm", [2, 3, 4, 5])
            self.write_dataset(test, "test", [2, 30, 100])
            calibration_out = root / "calibration-out"
            manifest = build_workload(
                dataset=calibration, warmup_dataset=train, output_dir=calibration_out,
                tokenizer=FakeTokenizer(), tokenizer_name="fake", tokenizer_revision="abc123",
                split="calibration", warmup_count=3,
            )
            records = [json.loads(line) for line in (calibration_out / "workload.jsonl").read_text().splitlines()]
            self.assertEqual({row["bucket"] for row in records}, {"short", "medium", "heavy"})
            self.assertNotIn("tags", records[0])
            self.assertNotIn("expected_tags", records[0])
            self.assertTrue((calibration_out / "SHA256SUMS").is_file())
            self.assertEqual(manifest["privacy"]["contains_expected_tags"], False)

            test_out = root / "test-out"
            test_manifest = build_workload(
                dataset=test, warmup_dataset=train, output_dir=test_out,
                tokenizer=FakeTokenizer(), tokenizer_name="fake", tokenizer_revision="abc123",
                split="test", threshold_manifest=calibration_out / "workload-manifest.json",
                warmup_count=2,
            )
            self.assertEqual(test_manifest["bucket_thresholds"], manifest["bucket_thresholds"])
            self.assertIsNotNone(test_manifest["bucket_threshold_source_manifest_sha256"])

    def test_rejects_test_without_calibration_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            test = root / "test.csv"
            train = root / "train.csv"
            self.write_dataset(test, "test", [1])
            self.write_dataset(train, "warm", [1, 2, 3])
            with self.assertRaisesRegex(ValueError, "threshold-manifest"):
                build_workload(
                    dataset=test, warmup_dataset=train, output_dir=root / "out",
                    tokenizer=FakeTokenizer(), tokenizer_name="fake", tokenizer_revision="abc123",
                    split="test",
                )


if __name__ == "__main__":
    unittest.main()
