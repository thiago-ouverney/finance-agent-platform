import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_gguf_variant import VARIANTS, render_variant  # noqa: E402
from run_gguf_matrix import (  # noqa: E402
    CELL_TARGETS,
    SCHEDULE,
    parse_make_vars,
    run_matrix,
    validate_cell_results,
)


class RenderVariantTests(unittest.TestCase):
    def make_inputs(self, root: Path, variant: str) -> tuple[Path, Path, Path]:
        gguf = root / f"{variant}.gguf"
        gguf.write_bytes(b"GGUF" + variant.encode() * 19)
        tokenizer = root / "tokenizer"
        tokenizer.mkdir(exist_ok=True)
        (tokenizer / "tokenizer.json").write_text("{}\n")
        spec = VARIANTS[variant]
        manifest = root / f"{variant}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "artifacts": {
                        spec["manifest_key"]: {
                            "filename": gguf.name,
                            "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                            "quantization": spec["quantization"],
                            "uses_imatrix": spec["imatrix"],
                        }
                    }
                }
            )
        )
        return gguf, tokenizer, manifest

    def test_renders_absolute_consistent_configs_and_unique_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aliases = set()
            for variant in VARIANTS:
                gguf, tokenizer, manifest = self.make_inputs(root, variant)
                profile_path = render_variant(
                    variant=variant,
                    gguf=gguf,
                    tokenizer=tokenizer,
                    output_dir=root / "generated",
                    source_manifest=manifest,
                )
                profile = json.loads(profile_path.read_text())
                aliases.add(profile["alias"])
                self.assertTrue(Path(profile["gguf_path"]).is_absolute())
                self.assertEqual(profile["gguf_sha256"], hashlib.sha256(gguf.read_bytes()).hexdigest())
                for runtime in ("vllm", "llama", "ollama"):
                    config = json.loads(Path(profile["files"][runtime]["config"]).read_text())
                    launch = json.loads(Path(profile["files"][runtime]["launch"]).read_text())
                    self.assertEqual(config["model"], profile["alias"])
                    self.assertIn(profile["gguf_sha256"], config["model_artifact"])
                    if runtime != "ollama":
                        self.assertIn(str(gguf.resolve()), launch)
                        self.assertIn(profile["alias"], launch)
            self.assertEqual(len(aliases), 3)

    def test_rejects_non_gguf_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.gguf"
            bad.write_bytes(b"NOPE")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            with self.assertRaisesRegex(ValueError, "Assinatura GGUF inválida"):
                render_variant(
                    variant="q8_0",
                    gguf=bad,
                    tokenizer=tokenizer,
                    output_dir=root / "generated",
                    source_manifest=root / "missing-manifest.json",
                )

    def test_source_manifest_must_match_variant_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gguf, tokenizer, _ = self.make_inputs(root, "q4_k_m_imatrix")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "q4_k_m_imatrix": {
                                "filename": gguf.name,
                                "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                                "quantization": "Q4_K_M",
                                "uses_imatrix": False,
                            }
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Manifesto diverge"):
                render_variant(
                    variant="q4_k_m_imatrix",
                    gguf=gguf,
                    tokenizer=tokenizer,
                    output_dir=root / "generated",
                    source_manifest=manifest,
                )


class MatrixRunnerTests(unittest.TestCase):
    def prepare_profiles(self, root: Path) -> Path:
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text("{}\n")
        profiles = root / "profiles"
        for variant in VARIANTS:
            gguf = root / f"{variant}.gguf"
            gguf.write_bytes(b"GGUF" + variant.encode() * 31)
            spec = VARIANTS[variant]
            manifest = root / f"{variant}.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            spec["manifest_key"]: {
                                "filename": gguf.name,
                                "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                                "quantization": spec["quantization"],
                                "uses_imatrix": spec["imatrix"],
                            }
                        }
                    }
                )
            )
            render_variant(
                variant=variant,
                gguf=gguf,
                tokenizer=tokenizer,
                output_dir=profiles,
                source_manifest=manifest,
            )
        return profiles

    def fake_make(self, root: Path) -> Path:
        executable = root / "fake-make"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "with open(os.environ['FAKE_MAKE_LOG'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "joined = ' '.join(sys.argv[1:])\n"
            "fail = os.environ.get('FAKE_FAIL_CELL', '')\n"
            "returncode = 9 if fail and fail in joined else 0\n"
            "targets = {'gguf-smoke-vllm': 'vllm', 'gguf-smoke-llama': 'llama.cpp', "
            "'gguf-smoke-ollama': 'ollama', 'gguf-base-vllm': 'vllm', "
            "'gguf-base-llama': 'llama.cpp', 'gguf-base-ollama': 'ollama'}\n"
            "target = next((item for item in sys.argv[1:] if item in targets), None)\n"
            "if target and returncode == 0 and not os.environ.get('FAKE_SKIP_RESULTS'):\n"
            "    values = dict(item.split('=', 1) for item in sys.argv[1:] if '=' in item)\n"
            "    config_key = {'vllm': 'VLLM_CONFIG', 'llama.cpp': 'LLAMA_CONFIG', "
            "'ollama': 'OLLAMA_CONFIG'}[targets[target]]\n"
            "    config = json.loads(pathlib.Path(values[config_key]).read_text())\n"
            "    out = pathlib.Path(values['GGUF_COMPARISON_RESULTS']) / targets[target] / "
            "('stamp-' + values['GGUF_VARIANT']) / values['GGUF_RESULT_NAME'] / 'json'\n"
            "    out.mkdir(parents=True, exist_ok=True)\n"
            "    (out / 'manifest.json').write_text(json.dumps({'status': 'complete', "
            "'smoke': target.startswith('gguf-smoke-'), 'config': config}))\n"
            "raise SystemExit(returncode)\n"
        )
        executable.chmod(0o755)
        return executable

    def test_runs_preparation_then_nine_cells_sequentially_and_continues_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            results = root / "results"
            with mock.patch.dict(
                os.environ,
                {
                    "FAKE_MAKE_LOG": str(make_log),
                    "FAKE_FAIL_CELL": "smoke-vllm q4_k_m_imatrix-never-matches",
                },
                clear=False,
            ):
                # The exact failure selector is set below to match target + unique alias.
                os.environ["FAKE_FAIL_CELL"] = "smoke-vllm"
                status = run_matrix(
                    mode="smoke",
                    profiles_dir=profiles,
                    workdir=root,
                    base_makefile=base_makefile,
                    results_root=results,
                    lock_file=results / ".matrix.lock",
                    make=str(fake_make),
                )
            self.assertEqual(status, 1)
            calls = [json.loads(line) for line in make_log.read_text().splitlines()]
            self.assertEqual(len(calls), 12)
            self.assertTrue(all("gguf-prepare-ollama" in call for call in calls[:3]))
            cell_calls = calls[3:]
            observed = []
            for call in cell_calls:
                target = next(value for value in call if value in CELL_TARGETS["smoke"].values())
                runtime = next(
                    name for name, target_name in CELL_TARGETS["smoke"].items()
                    if target_name == target
                )
                alias_arg = next(value for value in call if value.startswith("OLLAMA_MODEL_NAME="))
                alias = alias_arg.split("=", 1)[1]
                variant = next(key for key, spec in VARIANTS.items() if spec["alias"] == alias)
                observed.append((runtime, variant))
                path_arg = next(value for value in call if value.startswith("MODEL_7B_GGUF="))
                self.assertIn(variant, path_arg)
                profile = json.loads((profiles / variant / "profile.json").read_text())
                config_prefix = {"vllm": "VLLM_CONFIG=", "llama": "LLAMA_CONFIG=", "ollama": "OLLAMA_CONFIG="}[runtime]
                launch_prefix = {"vllm": "VLLM_LAUNCH=", "llama": "LLAMA_LAUNCH=", "ollama": "OLLAMA_LAUNCH="}[runtime]
                config_arg = next(value for value in call if value.startswith(config_prefix))
                launch_arg = next(value for value in call if value.startswith(launch_prefix))
                self.assertEqual(config_arg.split("=", 1)[1], profile["files"][runtime]["config"])
                self.assertEqual(launch_arg.split("=", 1)[1], profile["files"][runtime]["launch"])
                results_arg = next(
                    value for value in call if value.startswith("GGUF_COMPARISON_RESULTS=")
                )
                self.assertEqual(results_arg.split("=", 1)[1], str(results.resolve()))
            self.assertEqual(observed, list(SCHEDULE))
            # Every vLLM cell failed, but llama.cpp/Ollama cells after them still ran.
            campaign = next((results / "gguf-comparison").iterdir())
            manifest = json.loads((campaign / "campaign.json").read_text())
            self.assertEqual(len(manifest["cells"]), 9)
            self.assertEqual(manifest["summary"]["cells_total"], 9)
            self.assertGreater(manifest["summary"]["cells_failed_or_blocked"], 0)
            self.assertEqual(manifest["cells"][-1]["status"], "complete")

    def test_zero_exit_without_result_is_not_a_complete_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            results = root / "custom-results"
            with mock.patch.dict(
                os.environ,
                {"FAKE_MAKE_LOG": str(make_log), "FAKE_SKIP_RESULTS": "1"},
                clear=False,
            ):
                status = run_matrix(
                    mode="smoke",
                    profiles_dir=profiles,
                    workdir=root,
                    base_makefile=base_makefile,
                    results_root=results,
                    lock_file=root / "locks/matrix.lock",
                    make=str(fake_make),
                )

            self.assertEqual(status, 1)
            campaign = next((results / "gguf-comparison").iterdir())
            manifest = json.loads((campaign / "campaign.json").read_text())
            self.assertTrue(
                all(cell["status"] == "failed-invalid-result" for cell in manifest["cells"])
            )

    def test_global_lock_rejects_parallel_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            fake_make = self.fake_make(root)
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            results = root / "results"
            results.mkdir()
            lock_path = results / ".matrix.lock"
            with lock_path.open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status = run_matrix(
                    mode="smoke",
                    profiles_dir=profiles,
                    workdir=root,
                    base_makefile=base_makefile,
                    results_root=results,
                    lock_file=lock_path,
                    make=str(fake_make),
                )
            self.assertEqual(status, 2)

    def test_render_does_not_start_before_global_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            lock_path = root / "locks/matrix.lock"
            lock_path.parent.mkdir(parents=True)
            render_vars = [
                "GGUF_Q8_FILE=/tmp/q8.gguf",
                "GGUF_Q4_BASE_FILE=/tmp/q4.gguf",
                "GGUF_Q4_IMATRIX_FILE=/tmp/q4-imatrix.gguf",
                "GGUF_COMPARISON_MANIFEST=/tmp/manifest.json",
                "MODEL_7B_TOKENIZER=/tmp/tokenizer",
                "GGUF_COMPARISON_CONTEXT=16384",
                "GGUF_COMPARISON_GPU_MEMORY_UTILIZATION=0.95",
                "GGUF_COMPARISON_PYTHON=python3",
            ]
            with lock_path.open("a") as held, mock.patch.dict(
                os.environ,
                {"FAKE_MAKE_LOG": str(make_log)},
                clear=False,
            ):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status = run_matrix(
                    mode="smoke",
                    profiles_dir=profiles,
                    workdir=root,
                    base_makefile=base_makefile,
                    results_root=root / "results",
                    lock_file=lock_path,
                    make=str(fake_make),
                    render_target="gguf-comparison-render",
                    render_make_vars=render_vars,
                )

            self.assertEqual(status, 2)
            self.assertFalse(make_log.exists())

    def test_sha_change_is_rejected_before_any_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            profile = json.loads((profiles / "q8_0" / "profile.json").read_text())
            Path(profile["gguf_path"]).write_bytes(b"GGUFchanged")
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            with mock.patch.dict(os.environ, {"FAKE_MAKE_LOG": str(make_log)}, clear=False):
                with self.assertRaisesRegex(ValueError, "SHA-256 divergiu"):
                    run_matrix(
                        mode="formal",
                        profiles_dir=profiles,
                        workdir=root,
                        base_makefile=base_makefile,
                        results_root=root / "results",
                        lock_file=root / "results/.matrix.lock",
                        make=str(fake_make),
                    )
            self.assertFalse(make_log.exists())

    def test_tokenizer_change_is_rejected_before_any_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            (root / "tokenizer" / "tokenizer.json").write_text('{"changed":true}\n')
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            with mock.patch.dict(os.environ, {"FAKE_MAKE_LOG": str(make_log)}, clear=False):
                with self.assertRaisesRegex(ValueError, "Metadados do tokenizer divergiram"):
                    run_matrix(
                        mode="smoke",
                        profiles_dir=profiles,
                        workdir=root,
                        base_makefile=base_makefile,
                        results_root=root / "custom-results",
                        lock_file=root / "locks/matrix.lock",
                        make=str(fake_make),
                    )
            self.assertFalse(make_log.exists())

    def test_source_manifest_change_is_rejected_before_any_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = self.prepare_profiles(root)
            profile = json.loads((profiles / "q8_0" / "profile.json").read_text())
            source = Path(profile["source_manifest"]["path"])
            source.write_text(source.read_text() + "\n")
            fake_make = self.fake_make(root)
            make_log = root / "make.jsonl"
            base_makefile = root / "Makefile"
            base_makefile.write_text("all:\n\t@true\n")
            with mock.patch.dict(os.environ, {"FAKE_MAKE_LOG": str(make_log)}, clear=False):
                with self.assertRaisesRegex(ValueError, "Manifesto de origem divergiu"):
                    run_matrix(
                        mode="formal",
                        profiles_dir=profiles,
                        workdir=root,
                        base_makefile=base_makefile,
                        results_root=root / "results",
                        lock_file=root / "locks/matrix.lock",
                        make=str(fake_make),
                    )
            self.assertFalse(make_log.exists())

    def test_model_identity_cannot_be_overridden_with_extra_args(self):
        with self.assertRaisesRegex(ValueError, "não pode alterar identidade"):
            parse_make_vars(["LLAMA_EXTRA_ARGS=--model /tmp/outro.gguf"])
        with self.assertRaisesRegex(ValueError, "não pode alterar identidade"):
            parse_make_vars(["VLLM_EXTRA_ARGS=--model=/tmp/outro.gguf"])
        with self.assertRaisesRegex(ValueError, "não pode alterar identidade"):
            parse_make_vars(["LLAMA_EXTRA_ARGS=--alias=outro-modelo"])
        with self.assertRaisesRegex(ValueError, "não é permitida"):
            parse_make_vars(["HF_GGUF_REPO=outro/modelo"])

    def test_non_object_result_manifest_is_a_cell_validation_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timestamp = root / "vllm" / "stamp"
            manifest_path = timestamp / "run" / "json" / "manifest.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text("[]\n")
            profile = {
                "alias": "qwen25-7b-q8-0",
                "gguf_sha256": "a" * 64,
            }

            with self.assertRaisesRegex(ValueError, "deve ser um objeto JSON"):
                validate_cell_results(
                    [str(timestamp)],
                    profile=profile,
                    runtime="vllm",
                    mode="smoke",
                )

    def test_formal_matrix_selects_only_base_targets(self):
        self.assertEqual(
            CELL_TARGETS["formal"],
            {
                "vllm": "gguf-base-vllm",
                "llama": "gguf-base-llama",
                "ollama": "gguf-base-ollama",
            },
        )


class ComparisonMakefileTests(unittest.TestCase):
    def test_ollama_preparation_uses_matrix_context(self):
        result = subprocess.run(
            [
                "make",
                "-n",
                "-f",
                "Makefile.gguf-comparison",
                "-o",
                "prepare-benchmark",
                "gguf-prepare-ollama",
                "GGUF_COMPARISON_CONTEXT=4096",
                "OLLAMA_CONFIG=/tmp/config.json",
                "MODEL_7B_GGUF=/tmp/model.gguf",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--context "4096"', result.stdout)

    def test_render_target_creates_all_three_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            (tokenizer / "tokenizer.json").write_text("{}\n")
            artifacts = {}
            for variant in VARIANTS:
                artifact = root / f"{variant}.gguf"
                artifact.write_bytes(b"GGUF" + variant.encode() * 13)
                artifacts[variant] = artifact
            output = root / "generated"
            manifest = root / "gguf-comparison-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "artifacts": {
                            VARIANTS[variant]["manifest_key"]: {
                                "filename": artifact.name,
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                "quantization": VARIANTS[variant]["quantization"],
                                "uses_imatrix": VARIANTS[variant]["imatrix"],
                            }
                            for variant, artifact in artifacts.items()
                        }
                    }
                )
            )
            result = subprocess.run(
                [
                    "make",
                    "-f",
                    "Makefile.gguf-comparison",
                    "gguf-comparison-render",
                    f"GGUF_Q8_FILE={artifacts['q8_0']}",
                    f"GGUF_Q4_BASE_FILE={artifacts['q4_k_m_base']}",
                    f"GGUF_Q4_IMATRIX_FILE={artifacts['q4_k_m_imatrix']}",
                    f"MODEL_7B_TOKENIZER={tokenizer}",
                    f"GGUF_COMPARISON_OUTPUT={output}",
                    f"GGUF_COMPARISON_MANIFEST={manifest}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.parent.name for path in output.glob("*/profile.json")},
                set(VARIANTS),
            )

    def test_formal_cell_runs_base_benchmark_without_kv_sweep(self):
        result = subprocess.run(
            [
                "make",
                "-n",
                "-f",
                "Makefile.gguf-comparison",
                "-o",
                "prepare-vllm",
                "gguf-base-vllm",
                "GGUF_VARIANT=q4_k_m_imatrix",
                "VLLM_CONFIG=/tmp/config.json",
                "VLLM_LAUNCH=/tmp/launch.json",
                "VLLM_MODEL_DIR=/tmp/model.gguf",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bench.py run", result.stdout)
        self.assertIn('--result-name "gguf-q4_k_m_imatrix-base"', result.stdout)
        self.assertNotIn("run_kv_sweep.py", result.stdout)

    def test_smoke_cell_honors_custom_results_directory(self):
        result = subprocess.run(
            [
                "make",
                "-n",
                "-f",
                "Makefile.gguf-comparison",
                "-o",
                "prepare-vllm",
                "gguf-smoke-vllm",
                "GGUF_VARIANT=q8_0",
                "GGUF_COMPARISON_RESULTS=/tmp/gguf-custom-results",
                "VLLM_CONFIG=/tmp/config.json",
                "VLLM_LAUNCH=/tmp/launch.json",
                "VLLM_MODEL_DIR=/tmp/model.gguf",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--results "/tmp/gguf-custom-results"', result.stdout)


if __name__ == "__main__":
    unittest.main()
