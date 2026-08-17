"""
Evaluation framework for ticket analysis results.
Supports code-based evals and LLM-as-a-Judge evals for batch processing.
"""
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import asyncio

from pydantic import BaseModel, Field

from evaluators.llm_evals import LLMJudgeEvaluator
from evaluators.safety_evals import SafetyEvaluator
from evaluators.static_evals import StaticEvaluator
from evaluators.schema import EvalResult, EvalType

from llm import get_llm_client
from support_ticket.schema import SupportTicketBatchEvalResult, TicketAnalysis
from telemetry import PhoenixTelemetry

logger = logging.getLogger(__name__)

class SupportTicketStaticEvaluator(StaticEvaluator):
    """Rule-based code evaluations for ticket analysis."""
    
    @staticmethod
    def validate_category(analysis: TicketAnalysis) -> EvalResult:
        """Verify category is one of the valid options."""
        valid_categories = {
            "Billing/Payment", "Technical Support", "Account Management",
            "Product Inquiry", "Complaint", "Feature Request", "Other"
        }
        passed = analysis.category in valid_categories
        return EvalResult(
            eval_name="validate_category",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Category '{analysis.category}' is {'valid' if passed else 'invalid'}"
        )
    
    @staticmethod
    def validate_priority(analysis: TicketAnalysis) -> EvalResult:
        """Verify priority is one of the valid levels."""
        valid_priorities = {"critical", "high", "medium", "low"}
        passed = analysis.priority in valid_priorities
        return EvalResult(
            eval_name="validate_priority",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Priority '{analysis.priority}' is {'valid' if passed else 'invalid'}"
        )
    
    @staticmethod
    def validate_sentiment(analysis: TicketAnalysis) -> EvalResult:
        """Verify sentiment is one of the valid options."""
        valid_sentiments = {"positive", "neutral", "negative"}
        passed = analysis.sentiment in valid_sentiments
        return EvalResult(
            eval_name="validate_sentiment",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Sentiment '{analysis.sentiment}' is {'valid' if passed else 'invalid'}"
        )
    
    @staticmethod
    def check_intent_length(analysis: TicketAnalysis, min_length: int = 5) -> EvalResult:
        """Check intent description has reasonable length."""
        passed = len(analysis.intent) >= min_length
        return EvalResult(
            eval_name="check_intent_length",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Intent length ({len(analysis.intent)}) is {'sufficient' if passed else 'too short'}"
        )
    
    @staticmethod
    def check_summary_length(analysis: TicketAnalysis, min_length: int = 10) -> EvalResult:
        """Check summary has reasonable length."""
        passed = len(analysis.summary) >= min_length
        return EvalResult(
            eval_name="check_summary_length",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Summary length ({len(analysis.summary)}) is {'sufficient' if passed else 'too short'}"
        )
    
    @staticmethod
    def check_order_id_format(analysis: TicketAnalysis) -> EvalResult:
        """If order_id exists, check basic format (alphanumeric/numeric)."""
        if analysis.order_id is None:
            return EvalResult(
                eval_name="check_order_id_format",
                eval_type=EvalType.CODE,
                passed=True,
                score=1.0,
                reason="Order ID is null (valid)"
            )
        
        passed = analysis.order_id.replace("#", "").replace("-", "").isalnum()
        return EvalResult(
            eval_name="check_order_id_format",
            eval_type=EvalType.CODE,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"Order ID format is {'valid' if passed else 'invalid'}"
        )

class BatchEvaluator:
    """Orchestrates batch evaluation runs (suitable for cron jobs)."""
    
    def __init__(self, run_llm_judges: bool = True, judge_model: Optional[str] = None):
        """
        Initialize batch evaluator.
        
        Args:
            run_llm_judges: Whether to run LLM judge evaluations (slower)
            judge_model: Model to use for LLM judgments
        """
        if run_llm_judges and not judge_model:
            raise RuntimeError("Model required for LLM Judge")
        self.llm_judge_eval = LLMJudgeEvaluator(judge_model) if run_llm_judges else None
        self.safety_eval = SafetyEvaluator(judge_model) if run_llm_judges else None
        self.static_eval = SupportTicketStaticEvaluator()

    async def evaluate_ticket(
        self, ticket_id: str, message: str, analysis: TicketAnalysis
    ) -> SupportTicketBatchEvalResult:
        """
        Run all evaluations on a ticket analysis result.
        
        Args:
            ticket_id: Unique identifier for the ticket
            message: Original customer message
            analysis: Ticket analysis result
            
        Returns:
            BatchEvalResult with all evaluation results
        """
        eval_results: List[EvalResult] = []
        
        # Run static evaluations
        static_evals = [
            SupportTicketStaticEvaluator.validate_category(analysis),
            SupportTicketStaticEvaluator.validate_priority(analysis),
            SupportTicketStaticEvaluator.validate_sentiment(analysis),
            SupportTicketStaticEvaluator.check_intent_length(analysis),
            SupportTicketStaticEvaluator.check_summary_length(analysis),
            SupportTicketStaticEvaluator.check_order_id_format(analysis),
        ]
        eval_results.extend(eval_results)
        
        # Run LLM judge evaluations if enabled
        if self.llm_judge_eval:
            try:
                llm_evals = await asyncio.gather(
                    self.llm_judge_eval.evaluate_category_relevance(message, analysis),
                    self.llm_judge_eval.evaluate_sentiment_accuracy(message, analysis),
                    self.llm_judge_eval.evaluate_intent_clarity(message, analysis),
                    return_exceptions=True
                )
                # Filter out exceptions
                eval_results.extend([r for r in llm_evals if isinstance(r, EvalResult)])
            except Exception as e:
                logger.error(f"Error running LLM judge evaluations: {e}")
        
        return SupportTicketBatchEvalResult(
            ticket_id=ticket_id,
            message=message,
            analysis=analysis,
            eval_results=eval_results,
            timestamp=datetime.now(UTC).isoformat()
        )
    
    async def evaluate_batch(
        self,
        tickets: List[tuple[str, str, TicketAnalysis]]
    ) -> List[SupportTicketBatchEvalResult]:
        """
        Evaluate a batch of tickets.
        
        Args:
            tickets: List of (ticket_id, message, analysis) tuples
            
        Returns:
            List of BatchEvalResult for each ticket
        """
        logger.info(f"Starting batch evaluation of {len(tickets)} tickets")
        
        results = await asyncio.gather(
            *[
                self.evaluate_ticket(ticket_id, message, analysis)
                for ticket_id, message, analysis in tickets
            ],
            return_exceptions=True
        )
        
        # Filter out exceptions and log them
        eval_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation failed for ticket {i}: {result}")
            else:
                eval_results.append(result)
        
        logger.info(
            f"Batch evaluation complete. "
            f"Passed: {sum(1 for r in eval_results if r.all_passed)}/{len(eval_results)}"
        )
        
        return eval_results
