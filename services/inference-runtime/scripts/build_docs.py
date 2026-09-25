"""Gera HTML offline a partir dos dois documentos Markdown."""
from pathlib import Path
import os
import markdown

ROOT = Path(__file__).resolve().parents[1]
STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f3ec;color:#193835;font:17px/1.8 system-ui,sans-serif}main{max-width:1020px;margin:auto;padding:48px 28px}h1{font-size:clamp(36px,6vw,66px);line-height:1.1;letter-spacing:-1.8px}h2{font-size:28px;line-height:1.3;border-top:1px solid #cdd8ce;padding-top:32px;margin-top:44px;scroll-margin-top:20px}h3{font-size:22px}a{color:#136d58;text-underline-offset:4px}a:focus-visible{outline:3px solid #b54e27;outline-offset:3px}pre{overflow:auto;background:#193835;color:#f4f7ee;padding:22px;border-radius:12px;font-size:14px;line-height:1.6}code{font-family:ui-monospace,monospace;overflow-wrap:anywhere}p code,li code,td code{background:#e0e9df;padding:2px 5px;border-radius:4px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #cdd8ce;padding:12px;text-align:left;vertical-align:top}th{background:#e1eade}li{margin:9px 0}.table-wrap{overflow:auto}.eyebrow{font-size:12px;letter-spacing:2px;color:#136d58;font-weight:750}.toc{background:#fffdf7;border:1px solid #d3ddd0;padding:15px 25px;border-radius:12px;font-size:14px}.toc ul{list-style:none;padding-left:18px}blockquote{border-left:4px solid #b54e27;background:#fff0de;padding:8px 22px;margin:25px 0}footer{border-top:1px solid #cdd8ce;margin-top:40px;padding-top:20px;font-size:13px}.nav{display:flex;flex-wrap:wrap;gap:20px;font-size:14px}@media(max-width:650px){main{padding:25px 18px}body{font-size:16px}h2{font-size:25px}}@media print{body{background:white;font-size:11pt}h1{font-size:32pt}nav,.toc{display:none}pre{white-space:pre-wrap;color:black;background:#eee}.table-wrap{overflow:visible}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""
PAGES = [
    ("benchmark/metodologia", "Como funciona o benchmark"),
    ("benchmark/bench-explicado", "Bench explicado: base, replay e sweep"),
    ("engenharia/git-runpod", "Git no RunPod"),
    ("engenharia/codigo-explicado", "O código explicado em detalhes"),
    ("make/make-fluxo", "Fluxo completo do Makefile"),
]
for relative, title in PAGES:
    name = Path(relative).name
    section = Path(relative).parent
    source_path = ROOT / "docs" / f"{relative}.md"
    output_path = ROOT / "docs" / f"{relative}.html"
    source = source_path.read_text(encoding="utf-8")
    body = markdown.markdown("[TOC]\n\n" + source, extensions=["fenced_code", "tables", "toc"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    def link(path):
        return os.path.relpath(ROOT / path, output_path.parent).replace(os.sep, "/")
    page = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{STYLE}</style></head><body><main><div class="eyebrow">CHATBOT RUNTIME BENCH · GUIDELLM · UM USUÁRIO</div><nav class="nav"><a href="{link('docs/benchmark/bench-explicado.html')}">Bench explicado</a><a href="{link('docs/benchmark/metodologia.html')}">Método</a><a href="{link('docs/make/make-fluxo.html')}">Make</a><a href="{link('docs/engenharia/git-runpod.html')}">Git no pod</a><a href="{link('README.md')}">Instalação e execução</a></nav>{body}<footer>HTML autônomo, sem telemetria ou dependências externas. Fontes e limitações no texto. Editável em Markdown; gere novamente com python scripts/build_docs.py.</footer></main></body></html>'''
    output_path.write_text(page, encoding="utf-8")
    print(output_path)
