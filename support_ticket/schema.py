from typing import Literal

from pydantic import BaseModel, Field
from evaluators.schema import EvalResult

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

class SupportTicketBatchEvalResult(BaseModel):
    """Result of batch evaluation run."""
    ticket_id: str
    message: str
    analysis: TicketAnalysis
    eval_results: list[EvalResult]
    timestamp: str
    
    @property
    def all_passed(self) -> bool:
        """Check if all evaluations passed."""
        return all(result.passed for result in self.eval_results)
    
    @property
    def average_score(self) -> float:
        """Calculate average score across all evals."""
        if not self.eval_results:
            return 0.0
        return sum(r.score for r in self.eval_results) / len(self.eval_results)