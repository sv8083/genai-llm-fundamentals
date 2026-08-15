from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from support_ticket.api import router as support_router
from core.settings import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Minimal API",
    debug=settings.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(support_router, prefix="/ticket")
