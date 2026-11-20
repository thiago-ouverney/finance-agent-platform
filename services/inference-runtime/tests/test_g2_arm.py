import json
import tempfile
import unittest
from pathlib import Path

import g2_arm
import g2_cell


class ArmProgressTests(unittest.TestCase):
    def test_pending_skips_the_finished_cell(self):
        pending = g2_arm.pending_cells((1, 2, 4), 2, {(1, 1)})
        self.assertEqual(pending, [(1, 2), (2, 1), (2, 2), (4, 1), (4, 2)])

    def test_estimate_starts_from_the_prior_and_then_uses_the_average(self):
        average, left = g2_arm.estimate([], 26)
        self.assertEqual(average, 100.0)
        self.assertEqual(left, 2600.0)
        average, left = g2_arm.estimate([80.0, 120.0], 24)
        self.assertEqual(average, 100.0)
        self.assertEqual(left, 2400.0)

    def test_duration_format(self):
        self.assertEqual(g2_arm.format_duration(43), "43 s")
        self.assertEqual(g2_arm.format_duration(125), "2 min 05 s")

    def test_complete_cell_requires_a_clean_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.assertFalse(g2_arm.cell_is_complete(directory, 128))
            manifest = {
                "summary": {
                    "successful_count": 128,
                    "errored_count": 0,
                    "prompt_tokens_missing": 0,
                }
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(g2_arm.cell_is_complete(directory, 128))
            self.assertTrue(str(g2_cell.result_dir(1, 8192, True, 1, 1)).endswith("s1-r1"))

    def test_plan_accepts_only_frozen_slot_pairs(self):
        plan = g2_arm.parse_plan(["--parallel", "8", "--ctx-size", "65536"])
        self.assertEqual((plan.parallel, plan.ctx_size, plan.cache), (8, 65536, "on"))
        self.assertTrue(str(g2_arm.progress_log(8, 65536, True)).endswith("p8-ctx65536-cache-on-progress.log"))
        with self.assertRaises(SystemExit):
            g2_arm.parse_plan(["--parallel", "8", "--ctx-size", "8192"])

    def test_wait_port_free_times_out(self):
        import socket
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        self.assertFalse(g2_arm.wait_port_free("127.0.0.1", port, 0.5))
        sock.close()


if __name__ == "__main__":
    unittest.main()
