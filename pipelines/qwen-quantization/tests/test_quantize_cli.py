from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

import quantize  # noqa: E402


class QuantizeCliTest(unittest.TestCase):
    def run_in_temporary_directory(self, argv: list[str], *, golden: bool) -> object:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            argv = [value.replace("{output}", str(output)) for value in argv]

            def save_fake(*args: object, **kwargs: object) -> None:
                output.mkdir()

            summary = {
                "quantization_method": "gptq",
                "bits": 4,
                "source_model_path": "model",
            }
            previous = Path.cwd()
            os.chdir(root)
            try:
                golden_value = (["golden"], {"source_sha256": "hash"})
                with (
                    patch.object(quantize, "prepare_golden_calibration_dataset", return_value=golden_value) as prepare_golden,
                    patch.object(quantize, "prepare_calibration_dataset", return_value=["legacy"]) as prepare_legacy,
                    patch.object(quantize, "quantize_gptq", side_effect=save_fake) as quantize_gptq,
                    patch.object(quantize, "build_quantization_summary", return_value=summary),
                    patch.object(quantize, "_copy_license"),
                    patch.object(quantize, "_write_model_card"),
                ):
                    quantize.main(argv)
                    return prepare_golden, prepare_legacy, quantize_gptq
            finally:
                os.chdir(previous)

    def test_dispatches_golden_gptq_parameters(self) -> None:
        prepare_golden, prepare_legacy, quantize_gptq = self.run_in_temporary_directory(
            [
                "model",
                "{output}",
                "gptq",
                "4",
                "--calibration-csv",
                "train.csv",
                "--calibration-limit",
                "16",
                "--calibration-seed",
                "7",
                "--group-size",
                "64",
                "--batch-size",
                "2",
            ],
            golden=True,
        )

        prepare_golden.assert_called_once()
        prepare_legacy.assert_not_called()
        self.assertEqual(quantize_gptq.call_args.kwargs["calibration_dataset"], ["golden"])
        self.assertEqual(quantize_gptq.call_args.kwargs["group_size"], 64)
        self.assertEqual(quantize_gptq.call_args.kwargs["batch_size"], 2)

    def test_legacy_cli_keeps_mmlu_fallback(self) -> None:
        prepare_golden, prepare_legacy, quantize_gptq = self.run_in_temporary_directory(
            ["model", "{output}", "GPTQ", "4"],
            golden=False,
        )

        prepare_golden.assert_not_called()
        prepare_legacy.assert_called_once()
        self.assertEqual(quantize_gptq.call_args.kwargs["calibration_dataset"], ["legacy"])

    def test_awq_invalid_bits_fail_before_dataset_loading(self) -> None:
        with patch.object(quantize, "prepare_golden_calibration_dataset") as prepare:
            with self.assertRaisesRegex(ValueError, "only supports 4 bits"):
                quantize.main(["model", "output", "awq", "8", "--calibration-csv", "train.csv"])
        prepare.assert_not_called()

    def test_gptq_out_of_range_fails_before_dataset_loading(self) -> None:
        with patch.object(quantize, "prepare_calibration_dataset") as prepare:
            with self.assertRaisesRegex(ValueError, "GPTQ exige bits entre 2 e 8"):
                quantize.main(["model", "output", "gptq", "9"])
        prepare.assert_not_called()

    def test_rejects_already_quantized_source_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                '{"quantization_config": {"bits": 4}}', encoding="utf-8"
            )
            with patch.object(quantize, "prepare_calibration_dataset") as prepare:
                with self.assertRaisesRegex(ValueError, "dupla quantizacao"):
                    quantize.main([str(model), str(root / "output"), "gptq", "4"])
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
