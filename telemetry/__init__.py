"""
Phoenix telemetry and OpenTelemetry integration for LLM operations.
"""
import logging
from typing import Any, Optional
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

logger = logging.getLogger(__name__)


class PhoenixTelemetry:
    """Handles Phoenix OTLP telemetry setup and tracing."""
    
    _tracer_provider: Optional[TracerProvider] = None
    _tracer: Optional[Any] = None
    
    @classmethod
    def initialize(cls, endpoint: str, service_name: str = "support-ticket-service") -> bool:
        """
        Initialize Phoenix telemetry with OTLP exporter.
        
        Args:
            endpoint: Phoenix OTLP HTTP endpoint (e.g., http://localhost:6006/v1/traces)
            service_name: Service name for Phoenix
            
        Returns:
            bool: True if initialized successfully, False otherwise
        """
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available. Telemetry disabled.")
            return False
        
        try:
            # Create resource with service name
            resource = Resource.create({
                "service.name": service_name,
                "service.version": "1.0.0"
            })
            
            # Create OTLP exporter
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            
            # Create tracer provider with resource
            cls._tracer_provider = TracerProvider(resource=resource)
            cls._tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
            
            # Set global tracer provider
            trace.set_tracer_provider(cls._tracer_provider)
            
            # Get tracer
            cls._tracer = trace.get_tracer(__name__)
            
            logger.info(f"✓ Phoenix telemetry initialized at {endpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Phoenix telemetry: {e}")
            return False
    
    @classmethod
    @contextmanager
    def trace_llm_call(cls, operation_name: str, attributes: Optional[dict] = None):
        """
        Context manager for tracing LLM API calls.
        
        Args:
            operation_name: Name of the operation (e.g., "ticket_analysis")
            attributes: Optional dictionary of span attributes
            
        Yields:
            Span object for manual attribute updates
        """
        if cls._tracer is None:
            yield None
            return
        
        with cls._tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span
    
    @classmethod
    def shutdown(cls):
        """Shutdown telemetry and flush pending spans."""
        if cls._tracer_provider:
            cls._tracer_provider.force_flush()
            logger.info("Phoenix telemetry shutdown complete")


def get_telemetry() -> PhoenixTelemetry:
    """Get telemetry instance."""
    return PhoenixTelemetry
