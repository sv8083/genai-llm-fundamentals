from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from support_ticket.api import router as support_router
from telemetry import PhoenixTelemetry
from core.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    PhoenixTelemetry.initialize(
        endpoint=settings.phoenix_otlp_endpoint,
        service_name=settings.phoenix_project_name
    )
    yield
    # Shutdown
    PhoenixTelemetry.shutdown()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Minimal API",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(support_router, prefix="/ticket")
