import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
from lifecycle import Launch
from results_layout import prepare


class UnitTests(unittest.TestCase):
    def test_estimate_messages_tokens_counts_ids_not_batch_encoding_keys(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return {"input_ids": [[11, 12, 13, 14]], "attention_mask": [[1, 1, 1, 1]]}

        self.assertEqual(bench.estimate_messages_tokens(FakeTokenizer(), [{"role": "user", "content": "x"}]), 4)

    def test_synthetic_prompts_are_exact_reproducible_and_distinct(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

            def decode(self, ids, skip_special_tokens=False):
                return " ".join(ids)

        tokenizer = FakeTokenizer()
        first = bench._synthetic_prompt(tokenizer, 32, "42:0")
        repeated = bench._synthetic_prompt(tokenizer, 32, "42:0")
        second = bench._synthetic_prompt(tokenizer, 32, "42:1")

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(len(tokenizer.encode(first)), 32)
        self.assertEqual(len(tokenizer.encode(second)), 32)

    def test_independent_batch_uses_one_prompt_per_request(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

            def decode(self, ids, skip_special_tokens=False):
                return " ".join(ids)

        calls = []
        original = bench.stream_request

        def fake_stream(cfg, prompt, timeout, secret=""):
            calls.append(prompt)
            return {"request_args": json.dumps({"body": {"messages": [
                {"role": "user", "content": prompt}], "max_tokens": bench.OUTPUT_TOKENS}})}

        bench.stream_request = fake_stream
        bench.WORKLOADS["unit-independent"] = 24
        try:
            report = bench.run_stream_batch({}, FakeTokenizer(), "unit-independent", 3, 1, seed=42)
        finally:
            bench.stream_request = original
            bench.WORKLOADS.pop("unit-independent", None)

        self.assertEqual(len(set(calls)), 3)
        self.assertTrue(all(len(prompt.split()) == 24 for prompt in calls))
        rows = report["benchmarks"][0]["requests"]["successful"]
        self.assertEqual([row["workload_seed"] for row in rows], ["42:0", "42:1", "42:2"])

        warmup_calls = []
        bench.stream_request = lambda cfg, prompt, timeout, secret="": (
            warmup_calls.append(prompt) or {"request_args": json.dumps({"body": {
                "messages": [{"role": "user", "content": prompt}], "max_tokens": bench.OUTPUT_TOKENS}})})
        bench.WORKLOADS["unit-independent"] = 24
        try:
            bench.run_stream_batch({}, FakeTokenizer(), "unit-independent", 3, 1, seed=1_000_042)
        finally:
            bench.stream_request = original
            bench.WORKLOADS.pop("unit-independent", None)
        self.assertTrue(set(calls).isdisjoint(warmup_calls))

    def test_percentiles(self):
        self.assertEqual(bench.percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(bench.percentile([1, 2, 3], .95), 2.9)
        self.assertIsNone(bench.percentile([None], .95))

    def test_observability_rows_have_stable_unique_ids_without_prompt_or_output_contract(self):
        report = {"benchmarks": [{"requests": {
            "successful": [{"request_id": "short-1", "output": "privado",
                            "request_start_epoch_s": 10.0, "request_end_epoch_s": 11.0}],
            "errored": [{"request_id": "short-2", "error": "falha"}],
            "incomplete": [],
        }}]}
        rows = bench.observability_request_rows(
            report, experiment_id="exp", runtime="vllm", model="m",
            block="r1-short-measure", benchmark_phase="measure",
            scenario="short", repetition=1,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["request_uid"] for row in rows}), 2)
        self.assertEqual(rows[0]["benchmark_phase"], "measure")
        self.assertEqual(rows[1]["status"], "errored")

    def test_redaction(self):
        self.assertEqual(bench.redact({"api_key": "secret", "x": "a-secret"}, "secret"),
                         {"api_key": "[REDACTED]", "x": "a-[REDACTED]"})
        self.assertEqual(
            bench.redact(["server", "--api-key", "from-extra-args", "--token=inline"], ""),
            ["server", "--api-key", "[REDACTED]", "--token=[REDACTED]"],
        )
        with self.assertRaisesRegex(ValueError, "Segredos não podem"):
            bench.reject_sensitive_argv(["--api-key", "must-not-leak"])

    def test_model_metadata_uses_allowlist_and_redacts_environment_secrets(self):
        identity = bench.sanitize_model_identity({
            "schema_version": 1,
            "local_path": "/models/safe.gguf",
            "sha256": "a" * 64,
            "api_key": "must-not-leak",
            "artifact": {
                "kind": "gguf",
                "files": [{"path": "safe.gguf", "sha256": "a" * 64,
                           "bytes": 4, "authorization": "must-not-leak"}],
            },
        })
        encoded = json.dumps(identity)
        self.assertNotIn("must-not-leak", encoded)
        self.assertNotIn("api_key", encoded)
        self.assertNotIn("authorization", encoded)

    def test_vllm_runtime_environment_records_bundled_cuda_runtime(self):
        with mock.patch.dict(
            os.environ,
            {
                "VLLM_CUDA_RUNTIME_LIB": "/venv/site-packages/nvidia/cu13/lib",
                "LD_LIBRARY_PATH": "/must/not/be/persisted",
            },
            clear=True,
        ):
            environment = bench.runtime_environment({"runtime": "vllm"})
        self.assertEqual(environment, {
            "VLLM_CUDA_RUNTIME_LIB": "/venv/site-packages/nvidia/cu13/lib",
        })

    def test_cache_policy_separates_kv_prefix_and_residency(self):
        details = bench.describe_cache_policy(
            {"runtime": "vllm", "cache_policy": "recorded"},
            ["vllm", "serve", "--no-enable-prefix-caching", "--kv-cache-dtype", "fp8"],
            True,
        )
        self.assertEqual(details["prefix_cache"]["state"], "disabled-explicit-argv")
        self.assertEqual(details["model_residency"]["state"], "not-observed")
        self.assertEqual(details["effective_cache_argv"],
                         ["--no-enable-prefix-caching", "--kv-cache-dtype", "fp8"])
        self.assertEqual(details["prefix_cache"]["effective_argv_flags"],
                         ["--no-enable-prefix-caching"])
        self.assertIn("enabled-required", details["kv_cache"]["state"])

    def test_context_guard(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        bench.validate_run(cfg, ["short", "medium"], True)
        with self.assertRaises(ValueError):
            bench.validate_run(cfg, ["long"], True)
        with self.assertRaises(ValueError):
            bench.validate_run(cfg, ["short"], False)

        cfg["context_window"] = 8192
        bench.validate_run(cfg, ["short", "medium", "long"], True)
        self.assertEqual(bench.NAMED_WORKLOADS,
                         {"short": 256, "medium": 2048, "long": 7680})

    def test_context_guard_reports_observed_context_and_action(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        cfg["context_window"] = 2048
        with self.assertRaisesRegex(
            ValueError,
            r"medium exige context_window >= 2432, mas a configuração declara 2048.*OBS_CONTEXT=8192",
        ):
            bench.validate_run(cfg, ["medium"], True)

    def test_workload_contract_fingerprints_targets_and_prompt_authority(self):
        contract = bench.workload_contract(["short", "medium", "long"], "replay")
        self.assertEqual(contract["named_scenario_defaults"], bench.NAMED_WORKLOADS)
        self.assertEqual(contract["input_token_targets"], bench.NAMED_WORKLOADS)
        self.assertIn("truncation/padding", contract["prompt_policy"])
        self.assertIn("usage.prompt_tokens", contract["prompt_tokens_authority"])
        self.assertEqual(contract, bench.workload_contract(
            ["short", "medium", "long"], "replay"
        ))
        self.assertNotEqual(contract["sha256"],
                            bench.workload_contract(["short", "medium"], "replay")["sha256"])

    def test_profile_never_sweeps(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        spec = bench.scenario_config(cfg, "short", 30, 42, "", 300)["spec"]
        self.assertEqual(spec["profile"]["kind"], "synchronous")
        self.assertEqual(spec["data_loader"]["samples"], 30)
        self.assertEqual(spec["backend"]["extras"]["body"]["temperature"], 0)

    def test_url_guard(self):
        cfg = bench.load_config(bench.ROOT / "configs/vllm.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            cfg["base_url"] = "https://user:password@example.com"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                bench.load_config(path)

    def test_summary_excludes_errors(self):
        row = {"request_args": json.dumps({"body": {"messages": [{"role": "user", "content": "oi"}], "max_tokens": 128}}),
               "time_to_first_token_ms": 10, "request_latency": .5,
               "inter_token_latency_ms": 20, "output_tokens": 5, "prompt_tokens": 256}
        report = {"benchmarks": [{"requests": {"successful": [row], "errored": [{"request_latency": 100}], "incomplete": []}}]}
        out = bench.summarize(report)
        self.assertEqual(out["errored_request_count"], 1)
        self.assertEqual(out["request_latency_seconds_p50"], .5)
        self.assertTrue(out["percentiles_are_exploratory"])
        self.assertEqual(out["decode_generation_tokens_per_second_p50"], 50)
        self.assertEqual(out["effective_output_tokens_per_second_p50"], 10)
        self.assertEqual(out["request_first_token_latency_milliseconds_p99"], 10)

    def test_closed_loop_batch_accumulates_real_assistant_history(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()
            def decode(self, ids, skip_special_tokens=False):
                return " ".join(ids)
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                tokens = []
                for message in messages:
                    tokens.extend([message["role"], *message["content"].split()])
                if add_generation_prompt:
                    tokens.append("assistant")
                return tokens

        calls = []
        original = bench.stream_messages_request
        def fake_stream(cfg, messages, timeout, secret="", max_tokens=bench.OUTPUT_TOKENS, metadata=None):
            calls.append([dict(message) for message in messages])
            return {
                "request_start_time": 0.0, "first_token_time": 0.1, "request_end_time": 0.2,
                "time_to_first_token_seconds": 0.1, "generation_time_seconds": None,
                "end_to_end_latency_seconds": 0.2, "completion_tokens": 1,
                "prompt_tokens": len(messages) * 10, "total_tokens": len(messages) * 10 + 1,
                "decode_tokens_per_second": None, "end_to_end_tokens_per_second": 5,
                "inter_token_latency_seconds": None, "time_to_first_token_ms": 100,
                "request_latency": 0.2, "inter_token_latency_ms": None,
                "output_tokens": 1, "decode_tokens_s": None, "effective_tokens_s": 5,
                "stream_content_event_count": 1, "output": f"resposta {len(calls)}",
                "usage_observed": True, "request_sha256": f"hash-{len(calls)}",
                "request_args": json.dumps({"body": {"messages": messages, "max_tokens": max_tokens}}),
                **(metadata or {}), "history_tokens": len(messages) * 10,
            }
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "conversation.json"
            fixture_path.write_text(json.dumps({
                "name": "unit",
                "system": "sistema fixo",
                "turns": [{"user": "pergunta um"}, {"user": "pergunta dois"}],
            }))
            fixture = bench.load_conversation_fixture(fixture_path)
            bench.stream_messages_request = fake_stream
            try:
                bench.WORKLOADS["unit-conv"] = 80
                report = bench.run_conversational_batch({"model": "m"}, FakeTokenizer(), "unit-conv", 1, 2, 1,
                                                        fixture=fixture, loop_mode="closed-loop")
            finally:
                bench.stream_messages_request = original
                bench.WORKLOADS.pop("unit-conv", None)

        self.assertEqual(len(calls), 2)
        self.assertEqual([message["role"] for message in calls[0]], ["system", "user"])
        self.assertEqual([message["role"] for message in calls[1]], ["system", "user", "assistant", "user"])
        self.assertEqual(calls[1][2]["content"], "resposta 1")
        rows = report["benchmarks"][0]["requests"]["successful"]
        self.assertEqual([row["mode"] for row in rows], ["closed-loop", "closed-loop"])
        self.assertEqual([row["turn_index"] for row in rows], [1, 2])
        self.assertEqual(rows[0]["fixture_sha256"], fixture["sha256"])
        self.assertEqual(rows[1]["messages_count"], 4)
        summary = bench.summarize(report)
        self.assertEqual(summary["successful_request_count"], 2)
        self.assertEqual(summary["turn_indices"], [1, 2])
        self.assertEqual(bench.turn_manifest(report)[1]["fixture_sha256"], fixture["sha256"])
        self.assertEqual(bench.turn_manifest(report)[1]["history_tokens"], 40)
        self.assertIsInstance(summary["requests_sha256"], str)

    def test_replay_uses_fixed_assistant_from_fixture(self):
        fixture = {"sha256": "fixture-hash", "data": {"system": "s", "turns": [
            {"user": "u1", "assistant": "a1 fixa"},
            {"user": "u2", "assistant": "a2 fixa"},
        ]}}
        messages = bench.fixture_messages_for_turn(fixture, 2, ["resposta real ignorada"], "replay")
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "a1 fixa")

    def test_named_replay_scenarios_target_distinct_prompt_sizes(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()

            def decode(self, ids, skip_special_tokens=False):
                return " ".join(ids)

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
                tokens = []
                for message in messages:
                    tokens.extend([message["role"], *message["content"].split()])
                if add_generation_prompt:
                    tokens.append("assistant")
                return tokens

        tokenizer = FakeTokenizer()
        fixture = {"sha256": "fixture-hash", "data": {
            "name": "targeted-replay",
            "system": "sistema fixo",
            "turns": [
                {"user": f"pergunta número {index}",
                 "assistant": " ".join(f"a{index}-{word}" for word in range(150))}
                for index in range(1, 18)
            ],
        }}
        calls = []
        original = bench.stream_messages_request

        def fake_stream(cfg, messages, timeout, secret="", max_tokens=bench.OUTPUT_TOKENS,
                        metadata=None):
            estimate = bench.estimate_messages_tokens(tokenizer, messages)
            actual_prompt_tokens = estimate + 7
            calls.append([dict(message) for message in messages])
            return {
                **(metadata or {}),
                "prompt_tokens": actual_prompt_tokens,
                "history_tokens": actual_prompt_tokens,
                "output": "resposta",
            }

        bench.stream_messages_request = fake_stream
        try:
            rows = {}
            for scenario in ("short", "medium", "long"):
                report = bench.run_conversational_batch(
                    {"model": "m"}, tokenizer, scenario, 1, 1, 1,
                    fixture=fixture, loop_mode="replay",
                )
                rows[scenario] = report["benchmarks"][0]["requests"]["successful"][0]
        finally:
            bench.stream_messages_request = original

        estimates = [rows[name]["history_tokens_estimate"]
                     for name in ("short", "medium", "long")]
        self.assertEqual(len(set(estimates)), 3)
        self.assertEqual(len({rows[name]["turn_index"] for name in rows}), 3)
        for name, row in rows.items():
            target = bench.NAMED_WORKLOADS[name]
            self.assertLessEqual(row["history_tokens_estimate"], target)
            self.assertLessEqual(target - row["history_tokens_estimate"], 2)
            self.assertEqual(row["prompt_tokens"], row["history_tokens_estimate"] + 7)
            self.assertEqual(row["history_tokens"], row["prompt_tokens"])
            self.assertEqual(row["target_history_tokens"], target)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(messages[-1]["role"] == "user" for messages in calls))

        trimmed_messages, metadata = bench._target_replay_messages(
            tokenizer, fixture, 17, bench.NAMED_WORKLOADS["short"], "trim-test"
        )
        self.assertGreater(metadata["history_pairs_dropped"], 0)
        self.assertLessEqual(metadata["history_tokens_estimate"],
                             bench.NAMED_WORKLOADS["short"])
        self.assertIn("17", trimmed_messages[-1]["content"])

    def test_fixture_schema_and_replay_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversation.json"
            path.write_text(json.dumps({"turns": [{"user": "u1"}]}))
            with self.assertRaisesRegex(ValueError, "system"):
                bench.load_conversation_fixture(path)

            path.write_text(json.dumps({"system": "s", "turns": [{"user": "u1"}]}))
            fixture = bench.load_conversation_fixture(path)
            with self.assertRaisesRegex(ValueError, "replay exige assistant"):
                bench.validate_conversation_fixture_for_mode(fixture, "replay", 1)
            bench.validate_conversation_fixture_for_mode(fixture, "closed-loop", 1)

    def test_conversation_artifact_records_history_messages(self):
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        row = {
            "mode": "replay", "request_id": "short-c1-t1",
            "conversation_index": 1, "turn_index": 1,
            "request_args": json.dumps({"body": {"messages": messages, "max_tokens": 128}}),
            "output": "a", "request_sha256": "hash", "history_tokens": 12,
            "history_tokens_estimate": 10,
            "fixture_history_tokens_estimate": 18, "target_history_tokens": 12,
            "target_delta_tokens_estimate": 2, "history_pairs_dropped": 1,
            "synthetic_context_tokens": 3,
        }
        report = {"benchmarks": [{"requests": {"successful": [row], "errored": [], "incomplete": []}}]}
        artifact = bench.conversation_artifact(report, {"path": "fixture.json", "sha256": "fixture-hash",
                                                        "data": {"name": "unit", "version": 1}})
        self.assertEqual(artifact["fixture"]["sha256"], "fixture-hash")
        self.assertEqual(artifact["turns"][0]["messages"], messages)
        self.assertEqual(artifact["turns"][0]["assistant_output"], "a")
        self.assertEqual(artifact["turns"][0]["target_history_tokens"], 12)
        self.assertEqual(artifact["turns"][0]["history_pairs_dropped"], 1)

    def test_parser_allows_zero_warmup_only(self):
        original_argv, original_run = sys.argv, bench.run
        captured = {}

        def fake_run(args):
            captured["warmup"] = args.warmup
            captured["requests"] = args.requests

        try:
            bench.run = fake_run
            sys.argv = ["bench.py", "run", "--config", "cfg.json", "--local-model-path", "model.gguf",
                        "--requests", "1", "--repetitions", "1", "--warmup", "0"]
            bench.main()
            self.assertEqual(captured, {"warmup": 0, "requests": 1})

            sys.argv = ["bench.py", "run", "--config", "cfg.json", "--local-model-path", "model.gguf",
                        "--requests", "0", "--warmup", "0"]
            with self.assertRaises(SystemExit):
                bench.main()
        finally:
            bench.run = original_run
            sys.argv = original_argv

    def test_local_weights_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                bench.local_model_check(path)
            (path / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(bench.local_model_check(path)["bytes"], 7)
            (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Shards"):
                bench.local_model_check(path)

    def test_local_weights_rejects_empty_and_non_weight_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            empty = path / "empty.safetensors"
            empty.touch()
            with self.assertRaisesRegex(ValueError, "ausentes/vazios"):
                bench.local_model_check(empty)

            config = path / "config.json"
            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "arquivo de pesos"):
                bench.local_model_check(config)

    def test_launch_sets_huggingface_offline_environment(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as free:
            prepare(Path(tmp))
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
            env_file = Path(tmp) / "child-env.json"
            code = (
                "import json, os, pathlib; "
                f"pathlib.Path({str(env_file)!r}).write_text(json.dumps({{'HF_HUB_OFFLINE': os.getenv('HF_HUB_OFFLINE'), 'TRANSFORMERS_OFFLINE': os.getenv('TRANSFORMERS_OFFLINE'), 'HF_TOKEN': os.getenv('HF_TOKEN'), 'BENCH_API_KEY': os.getenv('BENCH_API_KEY')}})); "
                "import time; time.sleep(30)"
            )
            argv_file = Path(tmp) / "argv.json"
            argv_file.write_text(json.dumps([sys.executable, "-c", code]))
            launch = Launch({"base_url": f"http://127.0.0.1:{port}"}, argv_file, tmp)
            with mock.patch.dict("os.environ", {"HF_TOKEN": "private", "BENCH_API_KEY": "private"}):
                launch.start()
            try:
                deadline = time.monotonic() + 3
                while not env_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(env_file.exists())
                self.assertEqual(json.loads(env_file.read_text()), {
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "HF_TOKEN": None, "BENCH_API_KEY": None})
            finally:
                launch.close()

    def test_rate_edge_cases_and_bands(self):
        from reporting import derived, context_band, gpu_summary, ratio
        m = bench.stream_metrics(100.0, 100.5, 102.5, 103.0,
                                 {"prompt_tokens": 256, "completion_tokens": 4, "total_tokens": 260})
        self.assertAlmostEqual(m["time_to_first_token_seconds"], .5)
        self.assertAlmostEqual(m["generation_time_seconds"], 2.0)
        self.assertAlmostEqual(m["decode_tokens_per_second"], 2.0)
        self.assertAlmostEqual(m["end_to_end_tokens_per_second"], 4 / 3)
        # Uma janela periódica de log de 10 s não participa das fórmulas.
        m2 = bench.stream_metrics(100.0, 100.5, 102.5, 103.0,
                                   {"prompt_tokens": 256, "completion_tokens": 4, "total_tokens": 260})
        self.assertEqual(m["decode_tokens_per_second"], m2["decode_tokens_per_second"])
        missing = bench.stream_metrics(100.0, 100.5, 102.5, 103.0, None)
        self.assertIsNone(missing["completion_tokens"])
        self.assertIsNone(missing["decode_tokens_per_second"])
        one = bench.stream_metrics(100.0, 100.5, 100.5, 101.0,
                                   {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        self.assertIsNone(one["generation_time_seconds"])
        self.assertIsNone(one["inter_token_latency_seconds"])
        for n, itl in ((1, 10), (3, 0), (3, None), (3, float("nan"))):
            self.assertIsNone(derived({"output_tokens": n, "inter_token_latency_ms": itl})["decode_tokens_s"])
        self.assertIsNone(derived({"output_tokens": 2, "prompt_tokens": None,
                                   "request_latency": 0})["context_end_tokens"])
        self.assertIsNone(ratio(1, float("inf")))
        self.assertIsNone(ratio("1", 2))
        self.assertEqual(context_band(511), "[0, 512)")
        self.assertEqual(context_band(512), "[512, 1024)")
        self.assertEqual(context_band(2077), "[2048, 4096)")
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "gpu.csv").write_text(
                "phase,index,name,used_mib,total_mib,gpu_util_pct,temperature_c,power_w\n"
                "measure,0,GPU,N/A,,nan,N/A,bad\n")
            row = gpu_summary(tmp)[0]
            self.assertIsNone(row["used_mib_mean"])
            self.assertIsNone(row["gpu_util_pct_max"])

    def test_gpu_aggregation_and_html_escaping(self):
        from reporting import gpu_summary, render
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            prepare(p)
            (p / "gpu.csv").write_text("utc,phase,index,name,used_mib,total_mib,gpu_util_pct,temperature_c,power_w\n"
                "now,measure,0,<GPU>,10,24,50,60,N/A\nnow,measure,0,<GPU>,20,24,70,65,100\n"
                "now,warmup,0,<GPU>,5,24,20,50,80\nnow,measure,1,other,8,24,25,40,70\n")
            rows = gpu_summary(p)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["used_mib_max"], 20)
            self.assertEqual(rows[0]["gpu_util_pct_mean"], 60)
            self.assertEqual(rows[0]["power_w_mean"], 100)
            render(p, [])
            self.assertIn("&lt;GPU&gt;", (p / "html" / "summary.html").read_text())

    def test_context_estimate(self):
        from reporting import context_summary
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "r1-short-measure-requests.csv").write_text("status,prompt_tokens,output_tokens,inter_token_latency_ms,request_latency,time_to_first_token_ms\nsuccessful,256,128,20,3,460\nerrored,256,0,0,10,0\n")
            row = context_summary(p, 196608)[0]
            self.assertEqual(row["successful_request_count"], 1)
            self.assertEqual(row["scenario"], "short")
            self.assertEqual(row["estimated_start_logical_kv_cache_mebibytes"], 48)
            self.assertEqual(row["decode_generation_tokens_per_second_p50"], 50)


if __name__ == "__main__":
    unittest.main()
