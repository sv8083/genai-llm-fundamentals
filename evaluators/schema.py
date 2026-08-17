from pydantic import BaseModel, Field
from typing import Any

from enum import Enum

from support_ticket.schema import TicketAnalysis


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


class BatchEvalResult(BaseModel):
    ticket_id: str
    message: str
    analysis: TicketAnalysis # TODO make this genereic to support all ops
    eval_results: list[EvalResult]
    timestamp: str
    
    @property
    def all_passed(self) -> bool:
        return all(result.passed for result in self.eval_results)
    
    @property
    def average_score(self) -> float:
        if not self.eval_results:
            return 0.0
        return sum(r.score for r in self.eval_results) / len(self.eval_results)

class SafetyJudgeResult(BaseModel):
    safe: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str

class RubricJudgeResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
