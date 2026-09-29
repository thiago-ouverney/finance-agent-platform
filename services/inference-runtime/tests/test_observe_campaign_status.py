import contextlib
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts import observe_campaign_status as status


class ObserveCampaignStatusTests(unittest.TestCase):
    def write_run(self, root, runtime, stamp, *, smoke, state, profile="generic"):
        label = "smoke-short-medium-long" if smoke else "benchmark-short-medium-long"
        run_dir = root / runtime / stamp / label
        (run_dir / "json").mkdir(parents=True)
        (run_dir / "csv").mkdir()
        manifest = {
            "started_utc": stamp,
            "experiment_id": f"{stamp}-{runtime}-{label}",
            "config": {"runtime": "llama.cpp" if runtime == "llama" else runtime},
            "smoke": smoke,
            "status": state,
            "benchmark_profile": profile,
            "repetitions": 1,
            "scenarios": ["short", "medium", "long"],
            "blocks": [],
        }
        (run_dir / "json" / "manifest.json").write_text(json.dumps(manifest))
        with (run_dir / "csv" / "events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["phase", "event"])
            writer.writeheader()
            writer.writerow({"phase": "r1-short-measure", "event": "phase_change"})
        with (run_dir / "csv" / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "successful_request_count", "errored_request_count",
                "incomplete_request_count", "expected",
            ])
            writer.writeheader()
            writer.writerow({
                "successful_request_count": 3,
                "errored_request_count": 0,
                "incomplete_request_count": 0,
                "expected": 3,
            })
        return run_dir

    def test_discovers_latest_run_per_stage_and_renders_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_run(root, "vllm", "20260928T200000.000000Z", smoke=True, state="complete")
            latest = self.write_run(root, "vllm", "20260928T210000.000000Z", smoke=True, state="running")
            self.write_run(root, "llama", "20260928T211000.000000Z", smoke=True, state="complete")
            cutoff = datetime(2026, 9, 28, 19, tzinfo=timezone.utc)
            runs = status.discover(root, cutoff, "generic")
            self.assertEqual(runs[(True, "vllm")].run_dir, latest)
            self.assertEqual(runs[(True, "vllm")].phase, "r1-short-measure")
            rendered, complete, failed = status.render(root, cutoff, "generic")
            self.assertIn("smoke   vllm", rendered)
            self.assertIn("blocos=1/6", rendered)
            self.assertIn("req=3/3", rendered)
            self.assertFalse(complete)
            self.assertFalse(failed)

    def test_partial_json_is_ignored_and_failed_run_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            broken = root / "vllm" / "run" / "name" / "json"
            broken.mkdir(parents=True)
            (broken / "manifest.json").write_text("{")
            self.write_run(root, "ollama", "20260928T220000.000000Z", smoke=False, state="failed")
            cutoff = datetime(2026, 9, 28, 19, tzinfo=timezone.utc)
            with contextlib.redirect_stdout(io.StringIO()):
                rendered, complete, failed = status.render(root, cutoff, "generic")
            self.assertIn("formal ollama failed", " ".join(rendered.split()))
            self.assertFalse(complete)
            self.assertTrue(failed)
