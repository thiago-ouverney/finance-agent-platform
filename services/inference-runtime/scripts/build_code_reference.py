"""Apêndice Markdown com todas as linhas e índice de funções dos módulos centrais."""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parts = ["# Código completo: referência linha a linha\n\n"
         "Leia junto com [a explicação detalhada](codigo-explicado.md). "
         "Gerado dos arquivos reais; não é um segundo programa. Os números não pertencem ao Python.\n"]
for name in ("bench.py", "lifecycle.py", "reporting.py", "guidellm_compat.py"):
    source = (ROOT / name).read_text(encoding="utf-8")
    parts.append(f"\n## {name}\n\nSHA-256: `{hashlib.sha256(source.encode()).hexdigest()}`.\n\n")
    tree = ast.parse(source)
    parts.append("| Função/classe | Linhas |\n|---|---|\n")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            parts.append(f"| `{node.name}` | {node.lineno}–{node.end_lineno} |\n")
    parts.append("\n```text\n")
    parts.extend(f"{i:04d} | {line}".rstrip() + "\n" for i, line in enumerate(source.splitlines(), 1))
    parts.append("```\n")
(ROOT / "docs/engenharia/codigo-fontes.md").write_text("".join(parts), encoding="utf-8")
print("docs/engenharia/codigo-fontes.md atualizado")
