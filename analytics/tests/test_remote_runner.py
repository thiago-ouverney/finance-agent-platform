import unittest

from scripts.run_benchmark import make_arguments, remote_command


class RemoteRunnerTests(unittest.TestCase):
    def test_builds_quoted_remote_command(self):
        args = make_arguments("bench-ollama", ["MODEL_SIZE=7B", "BENCH_REQUESTS=10"])
        command = remote_command("/workspace/platform repo/runtime", args)
        self.assertIn("'/workspace/platform repo/runtime'", command)
        self.assertIn("BENCH_REQUESTS=10", command)

    def test_rejects_shell_fragments_as_variable_names(self):
        with self.assertRaises(ValueError):
            make_arguments("bench-all", ["bad;command=value"])


if __name__ == "__main__":
    unittest.main()
