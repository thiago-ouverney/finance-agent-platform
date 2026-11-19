from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ENTRYPOINTS = (
    "README.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/architecture.md",
    "specs/README.md",
    "specs/_template/spec.md",
    "specs/_template/implementation-plan.md",
    "specs/_template/tasks.md",
    "specs/_template/test-plan.md",
    "specs/_template/acceptance.md",
    "specs/_template/decisions.md",
    ".agents/skills/remote-workflow-safety/SKILL.md",
)

SCOPED_GUIDES = (
    "analytics/AGENTS.md",
    "apps/whatsapp-adapter/AGENTS.md",
    "docs/AGENTS.md",
    "pipelines/qwen-quantization/AGENTS.md",
    "services/inference-runtime/AGENTS.md",
)

SPEC_STATUSES = {"draft", "approved", "implemented", "superseded", "abandoned"}
HYPOTHESIS_OUTCOMES = {
    "not-applicable",
    "pending",
    "accepted",
    "rejected",
    "inconclusive",
}


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    try:
        block = text.split("---\n", 2)[1]
    except IndexError:
        return {}
    metadata: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata


class RepositoryGovernanceTest(unittest.TestCase):
    def test_required_governance_entrypoints_exist(self) -> None:
        missing = [path for path in REQUIRED_ENTRYPOINTS if not (ROOT / path).is_file()]
        self.assertEqual([], missing, f"Entradas de governança ausentes: {missing}")

    def test_risk_boundaries_have_scoped_guides(self) -> None:
        missing = [path for path in SCOPED_GUIDES if not (ROOT / path).is_file()]
        self.assertEqual([], missing, f"Guias escopados ausentes: {missing}")

        for relative_path in SCOPED_GUIDES:
            path = ROOT / relative_path
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            for section in ("## Papel e fronteira", "## Limites", "## Evidência"):
                self.assertIn(section, content, f"{relative_path} não contém {section}")

    def test_root_agent_guide_defines_the_operating_kernel(self) -> None:
        content = read("AGENTS.md")
        required_sections = (
            "## Precedência e fontes de verdade",
            "## Mapa do repositório",
            "## Fluxo de mudanças",
            "## Efeitos externos",
            "## Colaboração multiagente",
            "## Gates e conclusão",
        )
        missing = [section for section in required_sections if section not in content]
        self.assertEqual([], missing, f"Kernel operacional incompleto: {missing}")

    def test_specs_have_valid_metadata_and_one_durable_contract(self) -> None:
        specs_root = ROOT / "specs"
        self.assertTrue(specs_root.is_dir(), "Diretório specs/ ausente")
        if not specs_root.is_dir():
            return

        packs = [path for path in specs_root.iterdir() if path.is_dir() and path.name != "_template"]
        self.assertTrue(packs, "Nenhuma spec durável foi criada")
        for pack in packs:
            spec = pack / "spec.md"
            self.assertTrue(spec.is_file(), f"{pack.relative_to(ROOT)} não contém spec.md")
            if not spec.is_file():
                continue
            metadata = frontmatter(spec)
            self.assertEqual(pack.name, metadata.get("id"), f"id inválido em {spec}")
            self.assertIn(metadata.get("status"), SPEC_STATUSES, f"status inválido em {spec}")
            self.assertIn(
                metadata.get("hypothesis_outcome"),
                HYPOTHESIS_OUTCOMES,
                f"hypothesis_outcome inválido em {spec}",
            )
            self.assertTrue(metadata.get("owner"), f"owner ausente em {spec}")

        local_pack_files = [
            f"specs/example/{name}"
            for name in (
                "implementation-plan.md",
                "tasks.md",
                "test-plan.md",
                "acceptance.md",
                "decisions.md",
            )
        ]
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", *local_pack_files],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual(local_pack_files, ignored, "O pack temporal deve permanecer local")

        versioned_templates = [
            str(path.relative_to(ROOT)) for path in (ROOT / "specs/_template").glob("*.md")
        ]
        ignored_templates = subprocess.run(
            ["git", "check-ignore", "--no-index", *versioned_templates],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        self.assertEqual([], ignored_templates, "Templates necessários a um clone não podem ser ignorados")

    def test_local_skills_have_metadata_and_are_registered_once(self) -> None:
        skills_root = ROOT / ".agents" / "skills"
        self.assertTrue(skills_root.is_dir(), "Registro local de skills ausente")
        if not skills_root.is_dir():
            return

        root_guide = read("AGENTS.md")
        skill_files = sorted(skills_root.glob("*/SKILL.md"))
        self.assertTrue(skill_files, "Nenhuma skill local registrada")
        names: list[str] = []
        for skill_file in skill_files:
            metadata = frontmatter(skill_file)
            name = metadata.get("name", "")
            self.assertEqual(skill_file.parent.name, name, f"name inválido em {skill_file}")
            self.assertTrue(metadata.get("description"), f"description ausente em {skill_file}")
            names.append(name)
            self.assertEqual(
                1,
                root_guide.count(f"`{name}`"),
                f"A skill {name} deve ser registrada exatamente uma vez no AGENTS.md raiz",
            )
        self.assertEqual(len(names), len(set(names)), "Nomes de skills locais duplicados")

    def test_governance_links_resolve_locally(self) -> None:
        candidates = [
            ROOT / "README.md",
            ROOT / "docs" / "README.md",
            ROOT / "docs" / "architecture.md",
        ]
        candidates.extend(ROOT.rglob("AGENTS.md"))
        if (ROOT / "specs").exists():
            candidates.extend((ROOT / "specs").rglob("*.md"))

        broken: list[str] = []
        link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
        for document in sorted(set(candidates)):
            if not document.is_file():
                continue
            for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
                target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                relative_target = unquote(target.split("#", 1)[0])
                if not (document.parent / relative_target).resolve().exists():
                    broken.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual([], broken, "Links locais quebrados:\n" + "\n".join(broken))

    def test_generated_and_sensitive_paths_are_local_only(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        forbidden = [
            path
            for path in tracked
            if (
                (Path(path).name.startswith(".env") and not Path(path).name.endswith(".example"))
                or Path(path).suffix in {".gguf", ".safetensors", ".pem", ".key"}
                or ".baileys_auth" in Path(path).parts
                or path.startswith(("artifacts/", ".chart-data-", ".codex-deck-", ".codex-finalizer/"))
            )
        ]
        self.assertEqual([], forbidden, f"Arquivos sensíveis ou gerados rastreados: {forbidden}")

        local_only_examples = (
            ".env",
            "model.gguf",
            "analytics/data/private.csv",
            "analytics/results/run/output.json",
            "apps/whatsapp-adapter/.baileys_auth/creds.json",
            "artifacts/report.pptx",
            ".chart-data-example/candidate.pptx",
            ".codex-deck-example/rendered/slide.png",
            ".codex-finalizer/candidate.pptx",
        )
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", *local_only_examples],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        ignored = set(result.stdout.splitlines())
        missing = [path for path in local_only_examples if path not in ignored]
        self.assertEqual([], missing, f"Paths locais sem regra de ignore: {missing}")

    def test_root_makefile_exposes_and_runs_governance_gate(self) -> None:
        makefile = read("Makefile")
        targets = set(re.findall(r"^([A-Za-z0-9_.-]+):", makefile, flags=re.MULTILINE))
        expected = {"help", "setup", "test", "build", "verify", "test-governance"}
        self.assertEqual(set(), expected - targets, "Alvos Make documentados estão ausentes")
        test_line = re.search(r"^test:\s*(.*)$", makefile, flags=re.MULTILINE)
        self.assertIsNotNone(test_line, "Alvo test ausente")
        if test_line:
            self.assertIn("test-governance", test_line.group(1))

    def test_ci_workflows_live_at_repository_root(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        nested = [
            path
            for path in tracked
            if "/.github/workflows/" in path and (ROOT / path).is_file()
        ]
        self.assertEqual([], nested, f"Workflows fora da raiz não são descobertos pelo GitHub: {nested}")
        workflow = ROOT / ".github/workflows/ci.yml"
        self.assertTrue(workflow.is_file(), "Workflow raiz ausente")
        if workflow.is_file():
            content = workflow.read_text(encoding="utf-8")
            self.assertIn("make test-governance", content)
            self.assertIn('node-version: "20.19.0"', content)
            self.assertIn("npm run verify", content)


if __name__ == "__main__":
    unittest.main()
