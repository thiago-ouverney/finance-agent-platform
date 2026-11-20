from __future__ import annotations

import threading


class ResponseSwapper:
    """Defeito só de teste: troca o corpo de cada par de respostas concluídas.

    O caminho normal do gateway não usa esta classe. Ela existe para o teste
    provar que uma mistura entre agentes é visível.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._partner: tuple[threading.Event, dict[str, bytes | None]] | None = None

    def exchange(self, body: bytes) -> bytes:
        with self._lock:
            if self._partner is None:
                ready = threading.Event()
                box: dict[str, bytes | None] = {"mine": body, "theirs": None}
                self._partner = (ready, box)
            else:
                ready, box = self._partner
                box["theirs"] = body
                self._partner = None
                ready.set()
                mine = box["mine"]
                return body if mine is None else mine
        if not ready.wait(timeout=2):
            return body
        theirs = box["theirs"]
        return body if theirs is None else theirs
