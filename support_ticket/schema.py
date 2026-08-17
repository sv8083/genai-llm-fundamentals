from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str | None = Field(None, description="Model name to use (optional)")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="LLM top_p parameter (0.0-1.0)")


class TicketAnalyzeRequest(BaseModel):
    message: str
    model: ModelConfig | None = Field(None, description="Model configuration")


class TicketAnalysis(BaseModel):
    category: str
    intent: str
    priority: Literal["low", "medium", "high", "critical"]
    order_id: str | None = None
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str
