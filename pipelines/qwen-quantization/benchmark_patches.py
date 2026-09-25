from importlib import import_module


def patch_mmlu_redux_result_metadata() -> None:
    from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite

    mmlu_redux_module = import_module("evalution.benchmarks.mmlu_redux")

    def _fixed_result_metadata(self):
        resolved_subsets = self._resolved_subsets()
        metadata = BaseMultipleChoiceSuite.result_metadata(self)
        metadata.update(
            {
                "dataset_name": self.dataset_name,
                "subsets": list(resolved_subsets.canonicals),
                "subset_paths": [list(path) for path in resolved_subsets.paths],
                "subset_kinds": list(resolved_subsets.kinds),
                "selection_mode": resolved_subsets.selection_mode,
            }
        )
        return metadata

    mmlu_redux_module.MMLURedux.result_metadata = _fixed_result_metadata


def patch_ifeval_evaluate() -> None:
    from evalution.benchmarks.base import BaseTestSuite

    ifeval_module = import_module("evalution.benchmarks.ifeval")

    def _fixed_ifeval_evaluate(self, session):
        result = BaseTestSuite.evaluate(self, session)
        samples = list(result.samples)
        if not samples:
            return result

        prompt_level_strict_total = 0.0
        prompt_level_loose_total = 0.0
        strict_correct = 0.0
        strict_total = 0
        loose_correct = 0.0
        loose_total = 0

        for sample in samples:
            prompt_level_strict_total += 1.0 if ifeval_module._to_bool(sample.extracted.get("prompt_level_strict")) else 0.0
            prompt_level_loose_total += 1.0 if ifeval_module._to_bool(sample.extracted.get("prompt_level_loose")) else 0.0
            strict_values = [int(value) for value in sample.extracted.get("inst_level_strict", [])]
            loose_values = [int(value) for value in sample.extracted.get("inst_level_loose", [])]
            strict_correct += sum(strict_values)
            strict_total += len(strict_values)
            loose_correct += sum(loose_values)
            loose_total += len(loose_values)

        result.metrics = {
            "prompt_level_strict_acc": prompt_level_strict_total / len(samples),
            "prompt_level_loose_acc": prompt_level_loose_total / len(samples),
            "inst_level_strict_acc": strict_correct / (strict_total or 1),
            "inst_level_loose_acc": loose_correct / (loose_total or 1),
        }
        return result

    ifeval_module.IFEval.evaluate = _fixed_ifeval_evaluate