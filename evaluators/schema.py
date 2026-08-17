from pydantic import BaseModel, Field
from typing import Any, Literal

from enum import Enum

class EvalType(str, Enum):
    """Types of evaluations supported."""
    CODE = "static"  # Rule-based, static evaluations
    LLM_JUDGE = "llm_judge"  # LLM-as-a-Judge evaluations
    SAFETY = "safety"

class EvalResult(BaseModel):
    eval_name: str
    eval_type: EvalType
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None
    metadata: dict[str, Any] | None = None


# Here teporary
class TicketAnalysis(BaseModel):
    category: str
    intent: str
    priority: Literal["low", "medium", "high", "critical"]
    order_id: str | None = None
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str

class SafetyJudgeResult(BaseModel):
    safe: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str

class RubricJudgeResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
