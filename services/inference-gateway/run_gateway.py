#!/usr/bin/env python3
"""Sobe o gateway em 127.0.0.1 (padrão 8090) na frente do llama-server."""
from __future__ import annotations

import argparse

import uvicorn

from gateway.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-inflight", type=int, default=8)
    parser.add_argument("--max-queue", type=int, default=8)
    parser.add_argument("--retry-after", type=int, default=2)
    parser.add_argument("--queue-timeout", type=float, default=8)
    args = parser.parse_args()
    app = create_app(
        args.upstream,
        max_inflight=args.max_inflight,
        max_queue=args.max_queue,
        retry_after_s=args.retry_after,
        queue_timeout_s=args.queue_timeout,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
