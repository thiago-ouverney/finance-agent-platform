"""Renderiza o guia: python -m pip install Markdown==3.8.2; python build_html.py."""
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent
template = (ROOT / "template.html").read_text(encoding="utf-8")
for source, target in [("README.md", "index.html"), ("CHAT.md", "chat.html")]:
    md = markdown.Markdown(extensions=["fenced_code", "tables", "toc"], extension_configs={"toc": {"toc_depth": "2-2"}})
    body = md.convert((ROOT / source).read_text(encoding="utf-8"))
    body = body.replace('href="CHAT.md"', 'href="chat.html"')
    page = template.replace('href="README.md"', f'href="{source}"')
    (ROOT / target).write_text(page.replace("<!-- NAV -->", md.toc).replace("<!-- CONTENT -->", body), encoding="utf-8")
    print("HTML gerado:", ROOT / target)
