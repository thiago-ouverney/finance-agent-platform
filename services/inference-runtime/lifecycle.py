"""Cronometria de inicialização/primeiro stream, fora das fases GuideLLM.

Nenhum POST de inferência é enviado antes da primeira requisição medida.
O lançamento é opt-in e usa argv sem shell. Só o processo criado é encerrado.
"""
import json
import os
from pathlib import Path

from results_layout import artifact, href, prepare
import signal
import socket
import subprocess
import time
from urllib.parse import urlsplit

import httpx

DEFAULT_PROMPT = (
    "Explique para um estudante de estatística a diferença entre memória RAM e VRAM. "
    "Inclua um exemplo de uso de um chatbot e conclua com um resumo em três itens."
)


class Launch:
    def __init__(self, cfg, command_file, output, extra_args=None, executable=None):
        url = urlsplit(cfg["base_url"])
        if url.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("--launch só aceita servidor local (localhost).")
        self.host, self.port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
        self.argv = json.loads(Path(command_file).read_text(encoding="utf-8"))
        if not isinstance(self.argv, list) or not self.argv or any(not isinstance(x, str) or not x for x in self.argv):
            raise ValueError("O arquivo --launch deve conter um array JSON não vazio de strings (argv).")
        if executable:
            self.argv[0] = executable
        if extra_args:
            if any(not isinstance(x, str) or not x for x in extra_args):
                raise ValueError("--launch-extra-args aceita somente strings não vazias.")
            self.argv.extend(extra_args)
        self.output = Path(output)
        self.process = None
        self.log = None
        self.started = None

    def start(self):
        # Não interrompe servidores existentes nem tenta tomar uma porta ocupada.
        try:
            connection = socket.create_connection((self.host, self.port), timeout=1)
        except OSError:
            pass
        else:
            connection.close()
            raise ValueError("A porta já está em uso. Pare o seu servidor manualmente antes de usar --launch.")
        prepare(self.output)
        self.log = artifact(self.output, "server.log").open("w", encoding="utf-8")
        self.started = time.perf_counter()
        try:
            self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True, shell=False,
                                            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        except BaseException:
            self.log.close()
            raise
        return self.started

    def close(self):
        """Encerra exclusivamente o grupo criado por este objeto, inclusive filhos."""
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            # Alguns workers podem sobreviver ao encerramento do processo líder.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process is not None:
            # O líder pode ter falhado, mas workers/servidor podem ainda
            # estar no mesmo grupo. Tenta limpar o grupo sem falhar se ele já
            # tiver desaparecido.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.log is not None:
            self.log.close()
            self.log = None


def wait_models(cfg, secret, timeout, launch=None):
    """Apenas GET. Modelo listado não comprova que seus pesos estão na GPU."""
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    started = time.perf_counter()
    probes, last, next_update = 0, None, started + 15
    with httpx.Client(timeout=2, headers=headers, follow_redirects=False) as client:
        while True:
            probes += 1
            if launch is not None and launch.process.poll() is not None:
                raise RuntimeError(f"Servidor encerrou antes de ficar disponível (exit={launch.process.returncode}). Veja server.log.")
            try:
                response = client.get(cfg["base_url"] + "/v1/models")
                if response.status_code in {401, 403}:
                    raise ValueError("API recusou autenticação; confira BENCH_API_KEY.")
                response.raise_for_status()
                models = [m["id"] for m in response.json().get("data", [])]
                expected = cfg["model"]
                accepted = {expected}
                if cfg.get("runtime") == "ollama" and ":" not in expected:
                    accepted.add(expected + ":latest")
                if accepted.intersection(models):
                    observed = time.perf_counter()
                    return {"models": models, "get_probes": probes,
                            "wait_wall_s": observed - started,
                            "process_to_api_observed_s": observed - launch.started if launch else None,
                            "criterion": "GET /v1/models retornou o ID; não é prova de pesos residentes"}
                last = f"API respondeu, mas modelo esperado={expected!r}; disponíveis={models!r}"
                if models:
                    raise RuntimeError(last + ". Confira --alias/--served-model-name e config.model.")
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
                last = str(exc)
            elapsed = time.perf_counter() - started
            if not launch or elapsed >= timeout:
                raise RuntimeError(f"API/modelo não disponível após {elapsed:.1f}s: {last}")
            if time.perf_counter() >= next_update:
                print(f"[inicialização] runtime={cfg.get('runtime', 'não informado')} "
                      f"PID={launch.process.pid if launch else 'externo'} decorrido={elapsed:.1f}s "
                      f"limite={timeout}s; verificando GET {cfg['base_url']}/v1/models "
                      f"para modelo={cfg['model']!r}; última observação: {last}. "
                      f"Log do servidor: {artifact(launch.output, 'server.log') if launch else 'externo'}", flush=True)
                next_update = time.perf_counter() + 15
            time.sleep(min(.5, max(0, timeout - elapsed)))


def timed_request(cfg, secret, timeout, prompt, output, process_origin=None):
    """Primeiro POST é simultaneamente medição e validação, sem pré-aquecimento oculto.

    Grava resultado parcial inclusive em timeout, stream inválido ou usage ausente.
    TTFT aqui é primeiro conteúdo não vazio recebido (não mero cabeçalho/role).
    """
    path = Path(output)
    body = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 1, "max_tokens": 128, "stream": True,
            "stream_options": {"include_usage": True}}
    result = {"status": "running", "body": body, "stream_usage": None,
              "content_event_offsets_s": [], "output": "", "done": False,
              "ttft_ms": None, "e2e_s": None, "mean_itl_ms": None,
              "usage_observed": False, "request_start_time": None,
              "first_token_time": None, "request_end_time": None,
              "time_to_first_token_seconds": None, "generation_time_seconds": None,
              "end_to_end_latency_seconds": None, "completion_tokens": None,
              "prompt_tokens": None, "total_tokens": None,
              "decode_tokens_per_second": None, "end_to_end_tokens_per_second": None,
              "process_to_first_content_s": None, "process_to_response_end_s": None}
    headers = {"Authorization": f"Bearer {secret}"} if secret else {}
    started = None
    try:
        with httpx.Client(timeout=timeout, headers=headers, follow_redirects=False) as client:
            started = time.perf_counter()
            result["request_start_time"] = started
            with client.stream("POST", cfg["base_url"] + "/v1/chat/completions", json=body) as stream:
                result["headers_ms"] = (time.perf_counter() - started) * 1000
                stream.raise_for_status()
                if "text/event-stream" not in stream.headers.get("content-type", ""):
                    raise ValueError("A API não respondeu com SSE.")
                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(value)
                    if "error" in event:
                        raise ValueError(f"Erro no stream: {event['error']}")
                    result["stream_usage"] = event.get("usage") or result["stream_usage"]
                    text = "".join(c.get("delta", {}).get("content") or "" for c in event.get("choices", []))
                    if text:
                        now = time.perf_counter()
                        result["content_event_offsets_s"].append(now - started)
                        result["output"] += text
                        result["first_token_time"] = result["first_token_time"] or now
                        result["last_token_time"] = now
                        if result["ttft_ms"] is None:
                            result["ttft_ms"] = (now - started) * 1000
                            if process_origin is not None:
                                result["process_to_first_content_s"] = now - process_origin
            ended = time.perf_counter()
            result["request_end_time"] = ended
            result["e2e_s"] = ended - started
            result["end_to_end_latency_seconds"] = ended - started
            if process_origin is not None:
                result["process_to_response_end_s"] = ended - process_origin
            if not result["done"] or not result["output"]:
                raise ValueError("Stream incompleto ou sem conteúdo.")
            usage = result["stream_usage"]
            valid_usage = isinstance(usage, dict) and all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            result["usage_observed"] = valid_usage
            if valid_usage:
                result["completion_tokens"] = usage["completion_tokens"]
                result["prompt_tokens"] = usage["prompt_tokens"]
                result["total_tokens"] = usage["total_tokens"]
                result["time_to_first_token_seconds"] = ((result["first_token_time"] - started)
                                                           if result["first_token_time"] is not None else None)
                if result["completion_tokens"] > 1 and result.get("last_token_time") is not None:
                    result["generation_time_seconds"] = result["last_token_time"] - result["first_token_time"]
                    result["mean_itl_ms"] = 1000 * result["generation_time_seconds"] / (result["completion_tokens"] - 1)
                if result["generation_time_seconds"] and result["generation_time_seconds"] > 0:
                    result["decode_tokens_per_second"] = result["completion_tokens"] / result["generation_time_seconds"]
                if result["end_to_end_latency_seconds"] > 0:
                    result["end_to_end_tokens_per_second"] = result["completion_tokens"] / result["end_to_end_latency_seconds"]
                from reporting import derived
                result.update(derived({"output_tokens": usage["completion_tokens"], "prompt_tokens": usage["prompt_tokens"],
                                       "inter_token_latency_ms": result["mean_itl_ms"], "request_latency": result["e2e_s"]}))
            result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        result["error"] = str(exc)
        if started is not None:
            result["elapsed_until_exit_s"] = time.perf_counter() - started
        raise
    finally:
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        if secret:
            serialized = serialized.replace(secret, "[REDACTED]")
        path.write_text(serialized + "\n", encoding="utf-8")
    return result


def lifecycle_report(output, lifecycle):
    import html
    output = Path(output)
    artifact(output, "lifecycle.json").write_text(json.dumps(lifecycle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    readiness = lifecycle.get("readiness", {})
    rows.append(("Processo → API observada (s)", readiness.get("process_to_api_observed_s")))
    for phase in ("first_request", "warm_reference"):
        req = lifecycle.get(phase, {})
        for metric in ("ttft_ms", "decode_tokens_s", "effective_tokens_s", "e2e_s", "mean_itl_ms", "process_to_first_content_s", "process_to_response_end_s"):
            if phase == "warm_reference" and metric.startswith("process_"):
                continue
            rows.append((phase + " · " + metric, req.get(metric)))
    table = "".join(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value)) if value is not None else 'Não medido'}</td></tr>" for label, value in rows)
    page = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ciclo de vida</title><style>body{{font:17px/1.7 system-ui;background:#f6f3ec;color:#193835;margin:25px}}td,th{{padding:12px;border-bottom:1px solid #ccd6cc;text-align:left}}table{{width:100%;overflow-wrap:anywhere}}a{{color:#136d58}}</style><h1>Inicialização e primeira resposta</h1><p>Modo: {html.escape(lifecycle['mode'])}. Status: {html.escape(lifecycle['status'])}.</p><p>API disponível não implica modelo na GPU. O primeiro POST é cronometrado, sem teste de geração anterior. Processo novo não implica caches de disco/CUDA frios. A referência final repete o prompt e pode aproveitar prefix caching.</p><table>{table}</table><p><a href="{href('summary.html')}">Aquecimento e blocos GuideLLM</a> · <a href="{href('lifecycle.json')}">Dados do ciclo de vida</a></p></html>'''
    artifact(output, "lifecycle.html").write_text(page, encoding="utf-8")
