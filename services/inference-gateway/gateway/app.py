from __future__ import annotations

import httpx
from fastapi import Body, FastAPI, Header
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.admission import AdmissionController, AdmissionRejected
from gateway.metrics import render_metrics
from gateway.schema_check import assistant_text, failure_reason, request_schema
from gateway.swap import ResponseSwapper


def create_app(
    upstream_url: str,
    max_inflight: int = 8,
    max_queue: int = 8,
    retry_after_s: int = 2,
    queue_timeout_s: float = 8,
    swap_responses: bool = False,
) -> FastAPI:
    admission = AdmissionController(max_inflight, max_queue, retry_after_s, queue_timeout_s)
    swapper = ResponseSwapper() if swap_responses else None
    app = FastAPI()
    app.state.schema_failures = []
    app.state.rejections = {429: 0, 503: 0}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        body = render_metrics(
            admission.inflight,
            admission.queued,
            app.state.rejections[429],
            app.state.rejections[503],
            len(app.state.schema_failures),
        )
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/v1/chat/completions")
    def chat(
        payload: dict = Body(...),
        x_agent_id: str = Header(default="anonymous"),
    ) -> Response:
        try:
            token = admission.acquire(x_agent_id)
        except AdmissionRejected as exc:
            app.state.rejections[exc.status_code] = app.state.rejections.get(exc.status_code, 0) + 1
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"message": str(exc), "type": exc.error_type}},
                headers={"Retry-After": str(exc.retry_after_s)},
            )
        client = httpx.Client(timeout=60)
        try:
            response = client.send(
                client.build_request(
                    "POST",
                    f"{upstream_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                ),
                stream=True,
            )
        except Exception:
            client.close()
            token.release()
            raise

        def passthrough():
            chunks: list[bytes] = []
            delivered = b""
            try:
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    if swapper is None:
                        delivered += chunk
                        yield chunk
                if swapper is not None:
                    delivered = swapper.exchange(b"".join(chunks))
                    yield delivered
            finally:
                _record_schema(
                    app,
                    payload,
                    x_agent_id,
                    delivered if swapper is not None else b"".join(chunks),
                    response.headers.get("content-type", ""),
                )
                response.close()
                client.close()
                token.release()

        return StreamingResponse(
            passthrough(),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    return app


def _record_schema(app: FastAPI, payload: dict, agent_id: str, body: bytes, content_type: str) -> None:
    schema = request_schema(payload)
    if schema is None:
        return
    reason = failure_reason(schema, assistant_text(body, content_type))
    if reason is not None:
        app.state.schema_failures.append({"agent_id": agent_id, "reason": reason})
