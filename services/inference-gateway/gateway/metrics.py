from __future__ import annotations


def render_metrics(
    inflight: int,
    queued: int,
    rejected_429: int,
    rejected_503: int,
    schema_failures: int,
) -> str:
    return "\n".join(
        [
            "# TYPE gateway_inflight gauge",
            f"gateway_inflight {inflight}",
            "# TYPE gateway_queue gauge",
            f"gateway_queue {queued}",
            "# TYPE gateway_rejections_total counter",
            f'gateway_rejections_total{{status="429"}} {rejected_429}',
            f'gateway_rejections_total{{status="503"}} {rejected_503}',
            "# TYPE gateway_schema_failures_total counter",
            f"gateway_schema_failures_total {schema_failures}",
            "",
        ]
    )
