"""Métricas derivadas e relatório offline; não confunde contexto com VRAM."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from results_layout import artifact, href, locate


def ratio(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return a / b if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0 else None


def percentile(values, q):
    """Percentil empírico com interpolação linear; ausências não viram zero."""
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def derived(row):
    n, p = row.get("output_tokens"), row.get("prompt_tokens")
    itl = row.get("inter_token_latency_ms")
    return {
        "decode_tokens_s": ratio(1000, itl) if n is not None and n > 1 else None,
        "effective_tokens_s": ratio(n, row.get("request_latency")),
        "context_start_tokens": p,
        # Comprimento lógico final; não é o número exato de posições materializadas.
        "context_end_tokens": p + n if p is not None and n is not None else None,
        "context_band": context_band(p),
    }


def context_band(p):
    if p is None:
        return "não informado"
    for limit in (512, 1024, 2048, 4096, 8192, 16384):
        if p < limit:
            lower = 0 if limit == 512 else limit // 2
            return f"[{lower}, {limit})"
    return "[16384, +∞)"


def table(rows, columns):
    def fmt(v):
        if v is None:
            return "Não disponível"
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(fmt(r.get(k)))}</td>" for k, _ in columns) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def context_summary(output, kv_bytes_per_token=None):
    groups = defaultdict(list)
    csv_dir = Path(output) / "csv"
    files = csv_dir.glob("r*-*-requests.csv") if csv_dir.exists() else Path(output).glob("r*-*-requests.csv")
    for file in sorted(files):
        rep, scenario, phase, _ = file.stem.split("-", 3)
        with file.open() as handle:
            for row in csv.DictReader(handle):
                if row["status"] != "successful":
                    continue
                for key in ("prompt_tokens", "output_tokens", "inter_token_latency_ms", "request_latency", "time_to_first_token_ms"):
                    row[key] = float(row[key]) if row.get(key) else None
                row.update(derived(row))
                # Preservamos o cenário mesmo quando dois cenários caem na
                # mesma faixa de contexto. Isso evita que short/medium/long
                # apareçam como uma única população no JSON derivado.
                groups[(phase, rep, scenario, row["context_band"])].append(row)
    result = []
    for (phase, rep, scenario, band), rows in groups.items():
        entry = {"phase": phase, "repetition": rep, "scenario": scenario,
                 "context_band": band, "successful_request_count": len(rows)}
        context_metrics = {
            "decode_generation_tokens_per_second": "decode_tokens_s",
            "effective_output_tokens_per_second": "effective_tokens_s",
            "request_first_token_latency_milliseconds": "time_to_first_token_ms",
            "initial_context_input_token_count": "context_start_tokens",
            "final_logical_context_token_count": "context_end_tokens",
        }
        for label, key in context_metrics.items():
            values = [r[key] for r in rows if r[key] is not None and math.isfinite(r[key])]
            entry[label + "_sample_count"] = len(values)
            entry[label + "_p50"] = percentile(values, .50)
            entry[label + "_p95"] = percentile(values, .95)
            entry[label + "_p99"] = percentile(values, .99)
        result.append(entry)
        for edge in ("start", "end"):
            tokens_key = "initial_context_input_token_count_p50" if edge == "start" else "final_logical_context_token_count_p50"
            tokens = entry[tokens_key]
            entry[f"estimated_{edge}_logical_kv_cache_mebibytes"] = tokens * kv_bytes_per_token / 1048576 if tokens is not None and kv_bytes_per_token else None
    return result


def gpu_summary(output):
    groups = defaultdict(list)
    path = locate(Path(output), "gpu.csv")
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                groups[(row["phase"], row["index"], row["name"])].append(row)
    result = []
    for (phase, index, name), rows in groups.items():
        entry = {"phase": phase, "gpu": index, "name": name, "samples": len(rows)}
        for key in ("used_mib", "total_mib", "gpu_util_pct", "temperature_c", "power_w"):
            values = []
            for row in rows:
                try:
                    value = float(row[key])
                    if math.isfinite(value):
                        values.append(value)
                except (ValueError, TypeError, KeyError):
                    pass
            entry[key + "_mean"] = sum(values) / len(values) if values else None
            entry[key + "_max"] = max(values) if values else None
        result.append(entry)
    return result


def _timeseries(output):
    path = locate(Path(output), "gpu.csv")
    if not path.exists():
        return []
    with path.open() as handle:
        rows = []
        for row in csv.DictReader(handle):
            try:
                elapsed = float(row["elapsed_s"])
                used = float(row["used_mib"])
                util = float(row["gpu_util_pct"])
                if all(math.isfinite(value) for value in (elapsed, used, util)):
                    rows.append({"elapsed_s": elapsed, "used_mib": used, "gpu_util_pct": util})
            except (KeyError, TypeError, ValueError):
                continue
        return rows


def _chart(rows, key, title, ylabel, color, maximum=None):
    """Gera SVG autônomo para abrir no navegador sem dependências externas."""
    if not rows:
        return f'<p class="chart-empty">Sem amostras válidas para {html.escape(title.lower())}.</p>'
    width, height, pad = 920, 260, 42
    xmax = max(row["elapsed_s"] for row in rows) or 1.0
    ymax = maximum or max(row[key] for row in rows) or 1.0
    points = []
    for row in rows:
        x = pad + (width - 2 * pad) * row["elapsed_s"] / xmax
        y = height - pad - (height - 2 * pad) * row[key] / ymax
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    svg = (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
           f'<rect width="100%" height="100%" fill="#fbfaf5"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#789"/>'
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#789"/>'
           f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
           f'<text x="{pad}" y="20" fill="#193835">{html.escape(title)}</text>'
           f'<text x="{pad}" y="{height-8}" fill="#526">0 s</text>'
           f'<text x="{width-pad-55}" y="{height-8}" fill="#526">{xmax:.1f} s</text>'
           f'<text x="6" y="{pad+5}" fill="#526">{ymax:.0f}</text>'
           f'<text x="6" y="{height-pad+5}" fill="#526">0</text></svg>')
    return svg


def render(output, rows):
    output = Path(output)
    lifecycle_path, manifest_path = locate(output, "lifecycle.json"), locate(output, "manifest.json")
    lifecycle = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else {}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    kv_bytes = manifest.get("model_availability", {}).get("kv_bytes_per_token")
    context, gpu = context_summary(output, kv_bytes), gpu_summary(output)
    artifact(output, "context-summary.json").write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    artifact(output, "gpu-summary.json").write_text(json.dumps(gpu, ensure_ascii=False, indent=2) + "\n")
    startup = lifecycle.get("readiness", {}).get("process_to_api_observed_s")
    initial = []
    for key, label in (("first_request", "Primeira resposta"), ("warm_reference", "Referência final")):
        req = lifecycle.get(key, {})
        usage = req.get("stream_usage") or {}
        d = derived({"output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                     "inter_token_latency_ms": req.get("mean_itl_ms"), "request_latency": req.get("e2e_s")})
        initial.append({"phase": label, "ttft": req.get("ttft_ms"), "e2e": req.get("e2e_s"), **d})
    metrics = table(rows, [("phase", "Fase"), ("scenario", "Cenário"), ("repetition", "Repetição"),
        ("expected", "Requisições previstas"), ("successful_request_count", "Requisições bem-sucedidas"),
        ("errored_request_count", "Requisições com erro"), ("incomplete_request_count", "Requisições incompletas"),
        ("missing_request_count", "Requisições ausentes do relatório bruto"),
        ("time_to_first_token_milliseconds_p50", "Time To First Token p50 (ms)"),
        ("time_to_first_token_milliseconds_p95", "Time To First Token p95 (ms)"),
        ("time_to_first_token_milliseconds_p99", "Time To First Token p99 (ms)"),
        ("generation_time_seconds_p50", "Tempo de geração p50 (s)"),
        ("decode_tokens_per_second_p50", "Tokens/s de decodificação p50"),
        ("end_to_end_tokens_per_second_p50", "Tokens/s ponta a ponta p50"),
        ("end_to_end_latency_seconds_p50", "Latência ponta a ponta p50 (s)"),
        ("input_prompt_token_count_p50", "Tokens de entrada p50"),
        ("output_completion_token_count_p50", "Tokens de saída p50")])
    by_context = table(context, [("phase", "Fase"), ("repetition", "Repetição"), ("scenario", "Cenário"), ("context_band", "Faixa de entrada (tokens)"),
        ("successful_request_count", "Requisições bem-sucedidas"),
        ("initial_context_input_token_count_p50", "Tokens de entrada inicial p50"),
        ("final_logical_context_token_count_p50", "Tokens de contexto lógico final p50"),
        ("estimated_start_logical_kv_cache_mebibytes", "KV lógico inicial estimado (MiB)"),
        ("estimated_end_logical_kv_cache_mebibytes", "KV lógico final estimado (MiB)"),
        ("request_first_token_latency_milliseconds_p50", "Latência da requisição até primeiro token p50 (ms)"),
        ("decode_generation_tokens_per_second_p50", "Velocidade de geração p50 (tokens/s)"),
        ("effective_output_tokens_per_second_p50", "Velocidade efetiva p50 (tokens/s)")])
    hardware = table(gpu, [("phase", "Fase"), ("gpu", "GPU"), ("name", "Nome"), ("samples", "Amostras"),
        ("used_mib_max", "Memória máx. (MiB)"), ("total_mib_max", "Memória total (MiB)"),
        ("gpu_util_pct_mean", "Utilização média (%)"), ("gpu_util_pct_max", "Utilização máx. (%)"),
        ("temperature_c_max", "Temperatura máx. (°C)"), ("power_w_mean", "Potência média (W)"), ("power_w_max", "Potência máx. (W)")]) if gpu else "<p>Não disponível: nenhuma amostra NVIDIA válida. Em Apple/Metal este coletor não mede GPU; isso não significa utilização zero.</p>"
    kvfile = locate(output, "kv-cache.csv")
    kvrows = []
    if kvfile.exists():
        with kvfile.open() as handle:
            groups = defaultdict(list)
            for r in csv.DictReader(handle):
                groups[(r["phase"], r["series"])].append(float(r["fraction"]) * 100)
            kvrows = [{"phase": p, "series": s, "n": len(v), "mean": sum(v)/len(v), "max": max(v)} for (p, s), v in groups.items()]
    kv = table(kvrows, [("phase", "Fase"), ("series", "Série do servidor"), ("n", "Amostras"), ("mean", "Ocupação média (%)"), ("max", "Ocupação máx. (%)")]) if kvrows else "<p>Ocupação real de KV não disponível nesta execução. Não foi estimada a partir da VRAM.</p>"
    first = table(initial, [("phase", "Fase"), ("ttft", "TTFT (ms)"), ("decode_tokens_s", "Geração (tokens/s)"), ("effective_tokens_s", "Efetiva (tokens/s)"), ("e2e", "Total (s)")])
    policy = manifest.get("model_availability", {}).get("policy", "Execução anterior: veja o estado inicial; ausência de download não verificada por esta versão.")
    failure = f'<p class="note">Execução não concluída: {html.escape(str(manifest["error"]))}. Dados parciais não constituem uma bateria válida.</p>' if manifest.get("error") else ""
    series = _timeseries(output)
    memory_chart = _chart(series, "used_mib", "Memória da GPU ocupada (MiB)", "MiB", "#b54e27")
    util_chart = _chart(series, "gpu_util_pct", "Utilização computacional da GPU (%)", "%", "#136d58", 100)
    artifact(output, "telemetry-memory.svg").write_text(memory_chart, encoding="utf-8") if series else None
    artifact(output, "telemetry-gpu-util.svg").write_text(util_chart, encoding="utf-8") if series else None
    page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark · latência, geração e GPU</title>
<style>body{{font:16px/1.7 system-ui;margin:32px;background:#f6f3ec;color:#193835}}main{{max-width:1400px;margin:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border:1px solid #ccd6cc;text-align:left}}th{{background:#e0e9df}}.scroll{{overflow:auto}}h2{{margin-top:38px}}a{{color:#136d58}}.note{{padding:16px;background:#fff0de;border-left:4px solid #b54e27}}.chart{{width:100%;max-height:280px;border:1px solid #ccd6cc;margin:10px 0 22px}}.chart-empty{{padding:16px;background:#fff0de}}</style><main>
<h1>Um usuário · latência, geração e GPU</h1><p>Status: <strong>{html.escape(manifest.get('status', 'desconhecido'))}</strong>. {html.escape(policy)}</p>{failure}
<p class="note">TTFT = espera pelo primeiro token/conteúdo observado. Geração = (tokens de saída − 1)/(tempo entre primeiro e último token). Efetiva = tokens de saída/tempo total da requisição, incluindo TTFT. São taxas por requisição, não throughput agregado de usuários.</p>
<h2>1. Inicialização e primeira resposta</h2><p>Processo → API disponível: {html.escape(str(startup)) if startup is not None else 'não medido'} s. <a href="lifecycle.html">Ver ciclo de vida completo</a>.</p>{first}
<h2>2. Aquecimento e operação posterior</h2><p>Percentis entre requisições bem-sucedidas. Warmup e measure separados; p95 com menos de 100 sucessos é exploratório. Geração indisponível com menos de dois tokens ou intervalo não positivo.</p>{metrics}
<h2>3. Tokens/s por faixa de contexto — proxy da carga de KV</h2><p>Faixa definida pela entrada real, incluindo template, antes do decode. O contexto cresce durante a saída; mostramos também seu comprimento lógico final. Esta é uma comparação de velocidades médias de respostas iniciadas em cada faixa, não uma medição token a token dentro de faixas de ocupação física do cache.</p>{by_context}
<p>Para atenção completa, mantendo modelo, dtype de KV e uma sequência: KV lógico ≈ 2 × camadas × cabeças KV × dimensão da cabeça × bytes por elemento × tokens. Pesos 4/8 bits não determinam o dtype do KV. Blocos, reserva, prefix caching e sliding window impedem tratar essa fórmula como medição de VRAM. MiB estimados só aparecem com --kv-bytes-per-token informado e verificado pelo operador; caso contrário, ficam indisponíveis.</p>
<h2>4. GPU, CPU, RAM e SSD por fase</h2><p>O monitor amostra aproximadamente 1 vez/s e relaciona cada amostra a eventos/fases. GPU vem do nvidia-smi; CPU, RAM e I/O de disco vêm do host. Máximos amostrados podem perder picos e não há atribuição por processo. N/A é ausência de dado, não zero.</p><h3>Memória e utilização ao longo da execução</h3>{memory_chart}{util_chart}<p><a href="{href('telemetry-memory.svg')}">SVG da memória</a> · <a href="{href('telemetry-gpu-util.svg')}">SVG da GPU-util</a> · <a href="{href('telemetry-summary.json')}">Resumo de telemetria</a> · <a href="{href('events.csv')}">Eventos</a> · <a href="{href('system.csv')}">CPU/RAM/SSD</a></p>{hardware}
<h2>5. Ocupação real do pool KV — vLLM</h2><p>Coleta opcional de /metrics via --collect-kv-metrics. Percentual de blocos ocupados do pool, não percentual de VRAM nem bytes. Séries/engines separados. Amostragem e atualização do servidor podem perder transientes; não sincronizada por token.</p>{kv}
<p><a href="{href('summary.json')}">Resumo JSON</a> · <a href="{href('context-summary.json')}">Faixas JSON</a> · <a href="{href('gpu-summary.json')}">GPU JSON</a> · <a href="{href('manifest.json')}">Manifesto</a></p></main></html>'''
    artifact(output, "summary.html").write_text(page, encoding="utf-8")
