import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
PROFILE_MAKEFILE = ROOT / "Makefile.profiling"


def make_features():
    result = subprocess.run(
        ["make", "-f", str(PROFILE_MAKEFILE), "-pRrq", ":"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith(".FEATURES :="):
            return set(line.split()[2:])
    return set()


class MakefileRegressionTests(unittest.TestCase):
    def test_make_requires_oneshell_on_old_versions(self):
        if "oneshell" in make_features():
            self.skipTest("make já suporta .ONESHELL")
        result = subprocess.run(
            ["make", "-f", str(PROFILE_MAKEFILE), "help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".ONESHELL", result.stderr)

    def require_oneshell(self):
        if "oneshell" not in make_features():
            self.skipTest("GNU Make local não suporta .ONESHELL")

    def test_invalid_prepare_offline_value_is_rejected_by_make(self):
        self.require_oneshell()
        result = subprocess.run(
            ["make", "PREPARE_OFFLINE=2", "prepare-benchmark"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PREPARE_OFFLINE", result.stderr)

    def test_offline_prepare_dry_run_has_no_install_or_download(self):
        self.require_oneshell()
        result = subprocess.run(
            ["make", "-n", "PREPARE_OFFLINE=1", "prepare-benchmark"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("validando imports", result.stdout)
        self.assertNotIn("pip install", result.stdout)
        self.assertNotIn("hf download", result.stdout)
        self.assertNotIn("snapshot_download", result.stdout)

    def test_bench_vllm_preserves_quoted_extra_args_and_timeout(self):
        self.require_oneshell()
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            events = temp / "python-events.jsonl"
            fake_python = temp / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['PYTHON_EVENTS'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_python.chmod(0o755)
            fake_vllm = temp / "vllm"
            fake_vllm.write_text("#!/bin/sh\nexit 0\n")
            fake_vllm.chmod(0o755)
            env = os.environ.copy()
            env["PYTHON_EVENTS"] = str(events)
            result = subprocess.run(
                [
                    "make", "-o", "prepare-vllm", "bench-vllm",
                    f"PYTHON={fake_python}", f"VLLM_BIN={fake_vllm}",
                    "VLLM_EXTRA_ARGS=--gpu-memory-utilization '0.7 value'",
                    "BENCH_STARTUP_TIMEOUT=17", "BENCH_SCENARIOS=short",
                    "BENCH_REQUESTS=1", "BENCH_REPETITIONS=1", "BENCH_WARMUP=1",
                    "SWEEP_START=2048", "SWEEP_STEP=2048",
                    "SWEEP_MAX_CONTEXT=4096", "SWEEP_REQUESTS=1",
                    "SWEEP_REPETITIONS=1", "SWEEP_WARMUP=1",
                ],
                cwd=ROOT, env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in events.read_text().splitlines()]
            bench_calls = [call for call in calls if "bench.py" in call]
            self.assertGreaterEqual(len(bench_calls), 1)
            self.assertIn("--launch-extra-args-json", bench_calls[0])
            extra_index = bench_calls[0].index("--launch-extra-args-json")
            self.assertEqual(json.loads(bench_calls[0][extra_index + 1]),
                             ["--gpu-memory-utilization", "0.7 value"])
            self.assertIn("--startup-timeout", bench_calls[0])
            self.assertEqual(bench_calls[0][bench_calls[0].index("--startup-timeout") + 1], "17")
            sweep_calls = [call for call in calls if "scripts/run_kv_sweep.py" in call]
            self.assertEqual(len(sweep_calls), 1)
            self.assertEqual(sweep_calls[0][sweep_calls[0].index("--start") + 1], "2048")
            self.assertEqual(sweep_calls[0][sweep_calls[0].index("--step") + 1], "2048")
            self.assertEqual(sweep_calls[0][sweep_calls[0].index("--max-context") + 1], "4096")


class _ProfilingWrapperMixin:
    def run_wrapper(self, wrapper, profiler):
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            fake = temp / profiler
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['ARGS_FILE'], 'w') as stream:\n"
                "    json.dump(sys.argv[1:], stream)\n"
            )
            fake.chmod(0o755)
            args_file = temp / "args.json"
            env = os.environ.copy()
            env["PATH"] = f"{temp}{os.pathsep}{env['PATH']}"
            env["ARGS_FILE"] = str(args_file)
            env["PROFILE_OUTPUT"] = str(temp / "result")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / wrapper), "server", "--flag", "value"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(args_file.read_text())


class ProfilingWrapperTests(_ProfilingWrapperMixin, unittest.TestCase):
    def test_all_profiling_targets_keep_output_under_requested_directory(self):
        if "oneshell" not in make_features():
            self.skipTest("GNU Make local não suporta .ONESHELL")
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            results = temp / "profiling results"
            events = temp / "events.jsonl"
            bin_dir = temp / "bin"
            bin_dir.mkdir()
            for name, option in (("nsys", "--output"), ("ncu", "--export")):
                (bin_dir / name).write_text(
                    "#!/usr/bin/env python3\n"
                    "import json, os, sys\n"
                    f"option = {option!r}\n"
                    "args = sys.argv[1:]\n"
                    "path = args[args.index(option) + 1]\n"
                    "with open(os.environ['PROFILE_EVENTS'], 'a') as stream:\n"
                    "    stream.write(json.dumps({'option': option, 'path': path, 'args': args}) + '\\n')\n"
                    "open(path, 'w').close()\n"
                )
                (bin_dir / name).chmod(0o755)
            vllm = bin_dir / "vllm fake"
            llama = bin_dir / "llama server fake"
            for runtime in (vllm, llama):
                runtime.write_text("#!/bin/sh\nexit 0\n")
                runtime.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            env["PROFILE_EVENTS"] = str(events)
            common = {
                "PYTHON": os.sys.executable,
                "PROFILE_RESULTS_DIR": str(results),
                "VLLM_BIN": str(vllm),
                "LLAMA_SERVER_BIN": str(llama),
            }
            for target in (
                "profile-vllm-nsys", "profile-vllm-ncu",
                "profile-llama-nsys", "profile-llama-ncu",
            ):
                command = ["make", "-f", str(PROFILE_MAKEFILE), *
                           [f"{key}={value}" for key, value in common.items()], target]
                result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                        capture_output=True)
                self.assertEqual(result.returncode, 0, f"{target}: {result.stderr}")
            records = [json.loads(line) for line in events.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            for record in records:
                output = Path(record["path"])
                self.assertNotEqual(output, Path("/trace"))
                self.assertNotEqual(output, Path("/roofline"))
                self.assertTrue(output.parent.parent == results)
                self.assertTrue(output.exists())

    def test_nsys_wrapper_passes_trace_configuration_and_command(self):
        args = self.run_wrapper("nsys-wrapper.sh", "nsys")
        self.assertEqual(args[:6], [
            "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            "--trace-fork-before-exec=true", "--output",
        ])
        self.assertTrue(args[6].endswith("/result"))
        self.assertEqual(args[7:], ["server", "--flag", "value"])

    def test_ncu_wrapper_passes_roofline_configuration_and_command(self):
        args = self.run_wrapper("ncu-wrapper.sh", "ncu")
        self.assertEqual(args[:8], [
            "--target-processes", "all", "--set", "roofline", "--launch-skip",
            "0", "--launch-count", "1",
        ])
        self.assertEqual(args[8], "--export")
        self.assertTrue(args[9].endswith("/result"))
        self.assertEqual(args[10:], ["server", "--flag", "value"])


if __name__ == "__main__":
    unittest.main()
