from typing import Literal

from pydantic import BaseModel


class TicketAnalyzeRequest(BaseModel):
    message: str


class TicketAnalysis(BaseModel):
    category: str
    intent: str
    priority: Literal["low", "medium", "high", "critical"]
    order_id: str | None = None
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str
