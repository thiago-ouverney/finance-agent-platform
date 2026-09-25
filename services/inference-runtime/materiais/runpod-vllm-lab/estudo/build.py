"""Monta um HTML autônomo, sem dependências. Execute: python3 estudo/build.py."""
from pathlib import Path

root = Path(__file__).resolve().parent
page = (root / "conteudo.html").read_text()
page = page.replace("<!-- STYLE -->", "<style>" + (root / "style.css").read_text() + "</style>")
page = page.replace("<!-- SCRIPT -->", "<script>" + (root / "interacoes.js").read_text() + "</script>")
output = root.parent / "estudo.html"
output.write_text(page, encoding="utf-8")
print(output)
