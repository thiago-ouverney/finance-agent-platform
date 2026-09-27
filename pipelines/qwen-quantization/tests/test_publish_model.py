from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from publish_model import _check_upload_contents, parse_flag  # noqa: E402


class PublishModelTest(unittest.TestCase):
    def test_flags_are_strict(self) -> None:
        self.assertTrue(parse_flag("1", "flag"))
        self.assertFalse(parse_flag("0", "flag"))
        with self.assertRaisesRegex(ValueError, "deve ser 1 ou 0"):
            parse_flag("true", "flag")

    def test_allows_quant_log_but_rejects_private_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "quant_log.csv").write_text("layer,loss\n", encoding="utf-8")
            _check_upload_contents(root)
            (root / "train.csv").write_text("private\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Arquivos privados"):
                _check_upload_contents(root)

    def test_rejects_env_and_private_key_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".env.local").write_text("HF_TOKEN=secret\n", encoding="utf-8")
            (root / "deploy.pem").write_text("private key\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Arquivos privados"):
                _check_upload_contents(root)


if __name__ == "__main__":
    unittest.main()
