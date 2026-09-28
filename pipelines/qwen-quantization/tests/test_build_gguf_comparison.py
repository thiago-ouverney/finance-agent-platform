from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))

from build_gguf_comparison import (  # noqa: E402
    FINAL_ARTIFACTS,
    artifact_paths,
    build,
    sha256_file,
    verify,
)


CONVERTER = r'''#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
output = pathlib.Path(args[args.index("--outfile") + 1])
output.write_bytes(b"GGUF-BF16-from-one-conversion")
'''


IMATRIX = r'''#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
if "--show-statistics" in args:
    matrix = pathlib.Path(args[args.index("--in-file") + 1])
    if not matrix.read_bytes().startswith(b"GGUF"):
        raise SystemExit("invalid imatrix")
elif "-o" in args:
    output = pathlib.Path(args[args.index("-o") + 1])
    output.write_bytes(b"GGUF-imatrix-calibration")
else:
    raise SystemExit("unexpected llama-imatrix arguments")
'''


QUANTIZE = r'''#!/usr/bin/env python3
import pathlib
import sys

args = sys.argv[1:]
if "--imatrix" in args:
    matrix_index = args.index("--imatrix")
    remaining = args[matrix_index + 2:]
    source, output, quantization, threads = remaining
    marker = b"with-imatrix"
else:
    source, output, quantization, threads = args
    marker = b"plain"
if not pathlib.Path(source).read_bytes().startswith(b"GGUF"):
    raise SystemExit("invalid BF16 source")
pathlib.Path(output).write_bytes(b"GGUF-" + quantization.encode() + b"-" + marker)
'''


class GGUFComparisonTest(unittest.TestCase):
    def write_executable(self, path: Path, source: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)

    def make_fixture(self, root: Path) -> tuple[argparse.Namespace, dict[str, Path], str]:
        llama_cpp = root / "llama.cpp"
        self.write_executable(llama_cpp / "convert_hf_to_gguf.py", CONVERTER)
        self.write_executable(llama_cpp / "build" / "bin" / "llama-imatrix", IMATRIX)
        self.write_executable(llama_cpp / "build" / "bin" / "llama-quantize", QUANTIZE)
        subprocess.run(["git", "init", "-q", str(llama_cpp)], check=True)
        subprocess.run(
            ["git", "-C", str(llama_cpp), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(llama_cpp), "config", "user.name", "GGUF Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(llama_cpp), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(llama_cpp), "commit", "-qm", "fake llama tools"],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"],
            text=True,
        ).strip()

        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
            encoding="utf-8",
        )
        (model / "model-00001-of-00001.safetensors").write_bytes(b"fake-bf16-weights")
        (model / ".source_revision").write_text("hf-immutable-revision\n", encoding="utf-8")

        corpus = root / "mopep-train.txt"
        corpus.write_text(
            "<|im_start|>user\nmodelo de negócio<|im_end|>\n"
            "<|im_start|>assistant\n['serviço']<|im_end|>\n",
            encoding="utf-8",
        )
        corpus_manifest = root / "mopep-train.manifest.json"
        corpus_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_sha256": "golden-dataset-sha256",
                    "corpus_sha256": sha256_file(corpus),
                    "selection_sha256": "selection-sha256",
                    "prompt_sha256": "prompt-sha256",
                    "tokenizer_sha256": "tokenizer-sha256",
                    "selected_examples": 2,
                    "selection_seed": 42,
                    "includes_expected_answer": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        args = argparse.Namespace(
            output_dir=root / "gguf-output",
            prefix="Qwen2.5-7B-Instruct",
            verify_only=False,
            llama_cpp_dir=llama_cpp,
            llama_cpp_revision=revision,
            hf_model_dir=model,
            source_model_id="Qwen/Qwen2.5-7B-Instruct",
            corpus=corpus,
            corpus_manifest=corpus_manifest,
            python=Path(sys.executable),
            threads=3,
            gpu_layers=99,
            context=512,
            batch=512,
            ubatch=512,
        )
        return args, artifact_paths(args.output_dir, args.prefix), revision

    def command_rows(self, path: Path) -> list[list[str]]:
        return [json.loads(line)["argv"] for line in path.read_text(encoding="utf-8").splitlines()]

    @staticmethod
    def tool_name(command: list[str]) -> str:
        # Conversion is invoked as `python convert_hf_to_gguf.py ...`; the llama
        # tools are direct executables.
        return Path(command[1] if Path(command[0]).name.startswith("python") else command[0]).name

    def test_builds_three_artifacts_with_exact_calibration_commands_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, paths, revision = self.make_fixture(Path(temporary))
            manifest = build(args)
            rows = self.command_rows(paths["commands"])

            converter_rows = [row for row in rows if self.tool_name(row) == "convert_hf_to_gguf.py"]
            quantize_rows = [row for row in rows if self.tool_name(row) == "llama-quantize"]
            imatrix_build_rows = [
                row
                for row in rows
                if self.tool_name(row) == "llama-imatrix" and "-o" in row
            ]

            self.assertEqual(len(converter_rows), 1)
            self.assertEqual(len(imatrix_build_rows), 1)
            self.assertEqual(len(quantize_rows), 3)
            self.assertIn("--outtype", converter_rows[0])
            self.assertEqual(converter_rows[0][converter_rows[0].index("--outtype") + 1], "bf16")
            self.assertIn("--parse-special", imatrix_build_rows[0])
            self.assertIn("--no-ppl", imatrix_build_rows[0])

            commands_by_quantization = {
                next(item for item in row if item in {"Q8_0", "Q4_K_M"}) + (
                    "-imatrix" if "--imatrix" in row else "-plain"
                ): row
                for row in quantize_rows
            }
            self.assertNotIn("--imatrix", commands_by_quantization["Q8_0-plain"])
            self.assertNotIn("--imatrix", commands_by_quantization["Q4_K_M-plain"])
            calibrated = commands_by_quantization["Q4_K_M-imatrix"]
            self.assertEqual(
                Path(calibrated[calibrated.index("--imatrix") + 1]),
                paths["imatrix"],
            )

            self.assertEqual(manifest["inputs"]["llama_cpp_revision"], revision)
            self.assertEqual(
                manifest["inputs"]["dataset_sha256"],
                "golden-dataset-sha256",
            )
            self.assertEqual(set(manifest["artifacts"]), set(FINAL_ARTIFACTS))
            for name, (_, uses_imatrix) in FINAL_ARTIFACTS.items():
                path = paths[name]
                self.assertEqual(path.read_bytes()[:4], b"GGUF")
                self.assertEqual(manifest["artifacts"][name]["bytes"], path.stat().st_size)
                self.assertEqual(manifest["artifacts"][name]["sha256"], sha256_file(path))
                self.assertEqual(manifest["artifacts"][name]["uses_imatrix"], uses_imatrix)
            stored = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)

    def test_resume_does_not_repeat_conversion_or_quantization_and_verify_checks_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, paths, _ = self.make_fixture(Path(temporary))
            first = build(args)
            rows_before = self.command_rows(paths["commands"])
            second = build(args)
            rows_after = self.command_rows(paths["commands"])

            for tool in ("convert_hf_to_gguf.py", "llama-quantize"):
                before = sum(self.tool_name(row) == tool for row in rows_before)
                after = sum(self.tool_name(row) == tool for row in rows_after)
                self.assertEqual(after, before)
            imatrix_generations_before = sum(
                self.tool_name(row) == "llama-imatrix" and "-o" in row
                for row in rows_before
            )
            imatrix_generations_after = sum(
                self.tool_name(row) == "llama-imatrix" and "-o" in row
                for row in rows_after
            )
            self.assertEqual(imatrix_generations_after, imatrix_generations_before)
            self.assertEqual(first["artifacts"], second["artifacts"])

            verified = verify(
                argparse.Namespace(output_dir=args.output_dir, prefix=args.prefix)
            )
            self.assertEqual(verified, second)
            paths["q8_0"].write_bytes(paths["q8_0"].read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "Integridade diverg"):
                verify(argparse.Namespace(output_dir=args.output_dir, prefix=args.prefix))
            with self.assertRaisesRegex(ValueError, "Integridade diverg"):
                build(args)
            self.assertEqual(
                json.loads(paths["manifest"].read_text(encoding="utf-8")),
                second,
            )

    def test_custom_llama_build_directory_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, paths, _ = self.make_fixture(root)
            custom_build = root / "custom-llama-build"
            (args.llama_cpp_dir / "build").rename(custom_build)
            args.llama_cpp_build_dir = custom_build

            manifest = build(args)

            self.assertEqual(set(manifest["artifacts"]), set(FINAL_ARTIFACTS))
            rows = self.command_rows(paths["commands"])
            tool_paths = {
                Path(row[0]).resolve()
                for row in rows
                if self.tool_name(row) in {"llama-imatrix", "llama-quantize"}
            }
            self.assertTrue(tool_paths)
            self.assertTrue(all(custom_build.resolve() in path.parents for path in tool_paths))

    def test_resume_before_final_manifest_rejects_replaced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, paths, _ = self.make_fixture(Path(temporary))
            build(args)
            paths["manifest"].unlink()
            paths["q4_k_m"].write_bytes(b"GGUF-replaced-after-crash")

            with self.assertRaisesRegex(ValueError, "diverge do progresso"):
                build(args)

            self.assertFalse(paths["manifest"].exists())

    def test_resume_refuses_outputs_without_progress_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args, paths, _ = self.make_fixture(Path(temporary))
            build(args)
            paths["manifest"].unlink()
            paths["progress"].unlink()

            with self.assertRaisesRegex(ValueError, "sem manifesto de progresso"):
                build(args)

            self.assertFalse(paths["manifest"].exists())

    def test_rejects_llama_source_config_and_corpus_divergence(self) -> None:
        mutations = {
            "llama revision": lambda args, paths: setattr(
                args,
                "llama_cpp_revision",
                "0" * 40,
            ),
            "source revision": lambda args, paths: (
                args.hf_model_dir / ".source_revision"
            ).write_text("different-hf-revision\n", encoding="utf-8"),
            "source config": lambda args, paths: (
                args.hf_model_dir / "config.json"
            ).write_text(
                json.dumps({"model_type": "qwen2", "architectures": ["Changed"]}),
                encoding="utf-8",
            ),
            "corpus": lambda args, paths: args.corpus.write_text(
                "changed private corpus",
                encoding="utf-8",
            ),
        }
        expected = {
            "llama revision": "esperado",
            "source revision": "Entradas mudaram",
            "source config": "Entradas mudaram",
            "corpus": "Hash do corpus diverge",
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                args, paths, _ = self.make_fixture(Path(temporary))
                if label != "llama revision":
                    build(args)
                mutate(args, paths)
                with self.assertRaisesRegex(ValueError, expected[label]):
                    build(args)

    def test_rejects_existing_or_partial_artifacts_without_overwriting(self) -> None:
        cases = ("final", "manifest", "partial")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                args, paths, _ = self.make_fixture(Path(temporary))
                args.output_dir.mkdir(parents=True)
                if case == "final":
                    occupied = paths["q8_0"]
                    expected = "Artefatos existem sem manifesto de entradas"
                elif case == "manifest":
                    occupied = paths["manifest"]
                    expected = "Artefatos existem sem manifesto de entradas"
                else:
                    occupied = paths["bf16"].with_name(paths["bf16"].name + ".partial")
                    expected = "Saida parcial encontrada"
                occupied.write_bytes(b"preserve-me")

                with self.assertRaisesRegex(ValueError, expected):
                    build(args)

                self.assertEqual(occupied.read_bytes(), b"preserve-me")

    def test_make_dry_run_does_not_execute_locked_build_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace-must-not-be-created"
            result = subprocess.run(
                [
                    "make",
                    "-n",
                    "gguf-build-comparison",
                    f"WORKSPACE={workspace}",
                    f"QUANT_MODEL_DIR={workspace / 'model'}",
                    f"QUANT_DATASET={workspace / 'train.csv'}",
                    f"GGUF_OUTPUT_DIR={workspace / 'output'}",
                    f"GGUF_LOCK_FILE={workspace / 'locks/build.lock'}",
                ],
                cwd=PIPELINE_DIR,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("_gguf-build-comparison-locked", result.stdout)
            self.assertFalse(workspace.exists())

    def test_smoke_target_fixes_small_model_and_eight_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace-must-not-be-created"
            result = subprocess.run(
                [
                    "make",
                    "-n",
                    "gguf-smoke-build",
                    f"WORKSPACE={workspace}",
                    f"GGUF_SMOKE_OUTPUT_DIR={workspace / 'smoke-output'}",
                ],
                cwd=PIPELINE_DIR,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("QUANT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct", result.stdout)
            self.assertIn("GGUF_SAMPLES=8", result.stdout)
            self.assertIn(str(workspace / "smoke-output"), result.stdout)
            self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
