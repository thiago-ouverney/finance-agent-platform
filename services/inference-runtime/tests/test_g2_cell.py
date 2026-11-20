import tempfile
import unittest
from pathlib import Path

import g2_cell


class CellPlanTests(unittest.TestCase):
    def test_launch_adds_no_cache_only_when_disabled(self):
        enabled = g2_cell.cell_launch_args("llama-server", "model.gguf", 1, 8192, True)
        disabled = g2_cell.cell_launch_args("llama-server", "model.gguf", 8, 65536, False)
        self.assertNotIn("--no-cache-prompt", enabled)
        self.assertIn("--parallel", enabled)
        self.assertEqual(enabled[enabled.index("--parallel") + 1], "1")
        self.assertEqual(enabled[enabled.index("--ctx-size") + 1], "8192")
        self.assertIn("--no-cache-prompt", disabled)
        self.assertEqual(disabled[disabled.index("--parallel") + 1], "8")

    def test_summary_uses_the_measured_window(self):
        rows = [
            {"request_latency": 1.0, "time_to_first_token_ms": 100, "output_tokens": 10, "prompt_tokens": 5},
            {"request_latency": 3.0, "time_to_first_token_ms": 300, "output_tokens": 30, "prompt_tokens": 7},
        ]
        summary = g2_cell.cell_summary(rows, 0, 4.0)
        self.assertEqual(summary["requests_per_second"], 0.5)
        self.assertEqual(summary["output_tokens_per_second"], 10.0)
        self.assertEqual(summary["e2e_s_p50"], 2.0)
        self.assertEqual(summary["prompt_tokens_missing"], 0)

    def test_frozen_pairs_reject_a_mismatch(self):
        with self.assertRaises(SystemExit):
            g2_cell.parse_args(["--parallel", "1", "--ctx-size", "16384", "--streams", "1", "--repetition", "1"])

    def test_props_compare_slot_context_not_the_pool(self):
        self.assertTrue(g2_cell.props_match(1, 8192, [8192], [1]))
        self.assertTrue(g2_cell.props_match(8, 65536, [8192], [8]))
        self.assertTrue(g2_cell.props_match(16, 131072, [8192, 8192], [16]))
        self.assertFalse(g2_cell.props_match(8, 65536, [65536], [8]))
        self.assertFalse(g2_cell.props_match(8, 65536, [8192], [1]))

    def test_incomplete_directory_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "s1-r1"
            destination.mkdir()
            (destination / "server.log").write_text("subida recusada\n", encoding="utf-8")
            g2_cell.discard_incomplete(destination)
            self.assertFalse(destination.exists())
            destination.mkdir()
            (destination / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                g2_cell.discard_incomplete(destination)

    def test_result_path_names_the_cell(self):
        path = g2_cell.result_dir(1, 8192, True, 1, 1)
        self.assertTrue(str(path).endswith("results/g2/p1-ctx8192-cache-on/s1-r1"))


if __name__ == "__main__":
    unittest.main()
