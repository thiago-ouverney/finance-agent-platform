"""Carga closed-loop para a curva do runtime nu.

O benchmark oficial continua sequencial. Este módulo só limita quantos pedidos
iguais aos da baseline ficam em voo. Não chama o GuideLLM: a versão 0.7.4
remonta as mensagens e, com alvo de tokens de saída, envia ``ignore_eos``.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from threading import Semaphore, Thread

STREAMS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
REQUESTS_PER_REPETITION = 128
REPETITIONS = 3
SLOT_PLANS = (
    {"parallel": 1, "ctx_size": 8192, "context_per_request": 8192},
    {"parallel": 8, "ctx_size": 65536, "context_per_request": 8192},
    {"parallel": 16, "ctx_size": 131072, "context_per_request": 8192},
)


def cycle_records(records: list, count: int) -> list:
    """Repete a ordem congelada até completar ``count`` posições."""
    if count < 1:
        raise ValueError("count deve ser positivo")
    if not records:
        raise ValueError("a lista de requisições está vazia")
    return [records[index % len(records)] for index in range(count)]


def run_closed_loop(records: list, request: Callable, streams: int) -> dict:
    """Mantém no máximo ``streams`` chamadas de ``request`` ao mesmo tempo.

    ``request(record, index)`` devolve o resultado ou levanta. Uma falha fica
    registrada e as demais posições continuam. A ordem dos resultados é a ordem
    de entrada.
    """
    if streams < 1:
        raise ValueError("streams deve ser positivo")
    slots = Semaphore(streams)
    successful: list = [None] * len(records)
    errored: list = [None] * len(records)

    def run_one(index: int, record) -> None:
        with slots:
            try:
                successful[index] = request(record, index)
            except Exception as exc:
                errored[index] = exc

    workers = [
        Thread(target=run_one, args=(index, record), name=f"closed-loop-{index}")
        for index, record in enumerate(records)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    return {
        "successful": successful,
        "errored": errored,
        "successful_count": sum(item is not None for item in successful),
        "errored_count": sum(item is not None for item in errored),
    }


def load_jsonl(path: Path) -> list[dict]:
    """Lê o workload congelado. Cada linha precisa de mensagens e identificador."""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"linha {line_number} não é um objeto")
        messages = record.get("initial_messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"linha {line_number} não tem initial_messages")
        if not record.get("request_id"):
            raise ValueError(f"linha {line_number} não tem request_id")
        records.append(record)
    if not records:
        raise ValueError(f"nenhuma requisição em {path}")
    return records


def collect_scalars(payload, key: str) -> list:
    """Junta os valores escalares de uma chave, em ordem de leitura."""
    found = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if key in node and not isinstance(node[key], (dict, list)):
                found.append(node[key])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found
