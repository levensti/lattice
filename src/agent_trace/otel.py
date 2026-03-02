from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Generator

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None  # type: ignore[assignment]

_tracer = None


def setup_tracer(service_name: str, endpoint: str | None = None) -> None:
    """Configure the OpenTelemetry tracer provider."""
    global _tracer
    if not HAS_OTEL:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(endpoint=endpoint)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("agent-trace")


def get_tracer():
    global _tracer
    if not HAS_OTEL:
        return None
    if _tracer is None:
        _tracer = trace.get_tracer("agent-trace")
    return _tracer


@contextmanager
def traced_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Generator:
    """Create an OTel span if available, otherwise a no-op context."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, str(v)[:4096])
        yield span
