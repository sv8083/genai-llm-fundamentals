
from typing import Any

from loguru import logger

from evaluators.schema import EvalResult, EvalType, RubricJudgeResult
from prompts.load import PromptLoader
from llm import get_llm_client
from support_ticket.schema import TicketAnalysis
from telemetry import PhoenixTelemetry


class LLMJudgeEvaluator:
    """Rubric-based LLM-as-a-Judge evaluations."""

    def __init__(self, judge_model: str | None = None):
        self.client = get_llm_client(model=judge_model)

    async def evaluate_category_relevance(
        self,
        message: str,
        analysis: TicketAnalysis,
    ) -> EvalResult:
        """Evaluate category relevance using rubric from file."""
        prompt = PromptLoader.render(
            "evals/category_rubric",
            message=message,
            analysis=analysis
        )

        try:
            with PhoenixTelemetry.trace_llm_call(
                "llm_judge_category_relevance",
                attributes={
                    "category": analysis.category,
                    "eval_type": "rubric"
                }
            ):
                result = self.client.generate(prompt)

            judgement = self._parse_judgement(result)

            return EvalResult(
                eval_name="evaluate_category_relevance",
                eval_type=EvalType.LLM_JUDGE,
                passed=judgement.passed,
                score=judgement.score,
                reason=(
                    f"{judgement.reason} "
                    f"(judge confidence={judgement.confidence:.2f})"
                ),
            )

        except Exception as e:
            logger.exception("LLM Judge evaluation failed")

            return EvalResult(
                eval_name="evaluate_category_relevance",
                eval_type=EvalType.LLM_JUDGE,
                passed=False,
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    async def evaluate_sentiment_accuracy(
        self,
        message: str,
        analysis: TicketAnalysis,
    ) -> EvalResult:
        """Evaluate sentiment accuracy using rubric from file."""
        prompt = PromptLoader.render(
            "evals/sentiment_rubric",
            message=message,
            sentiment=analysis.sentiment
        )

        try:
            with PhoenixTelemetry.trace_llm_call(
                "llm_judge_sentiment_accuracy",
                attributes={
                    "sentiment": analysis.sentiment,
                    "eval_type": "rubric"
                }
            ):
                result = self.client.generate(prompt)

            judgement = self._parse_judgement(result)

            return EvalResult(
                eval_name="evaluate_sentiment_accuracy",
                eval_type=EvalType.LLM_JUDGE,
                passed=judgement.passed,
                score=judgement.score,
                reason=(
                    f"{judgement.reason} "
                    f"(judge confidence={judgement.confidence:.2f})"
                ),
            )

        except Exception as e:
            logger.exception("LLM Judge evaluation failed")

            return EvalResult(
                eval_name="evaluate_sentiment_accuracy",
                eval_type=EvalType.LLM_JUDGE,
                passed=False,
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    async def evaluate_intent_clarity(
        self,
        message: str,
        analysis: TicketAnalysis,
    ) -> EvalResult:
        """Evaluate intent clarity using rubric from file."""
        prompt = PromptLoader.render(
            "evals/intent_rubric",
            message=message,
            intent=analysis.intent
        )

        try:
            with PhoenixTelemetry.trace_llm_call(
                "llm_judge_intent_clarity",
                attributes={
                    "intent": analysis.intent,
                    "eval_type": "rubric"
                }
            ):
                result = self.client.generate(prompt)

            judgement = self._parse_judgement(result)

            return EvalResult(
                eval_name="evaluate_intent_clarity",
                eval_type=EvalType.LLM_JUDGE,
                passed=judgement.passed,
                score=judgement.score,
                reason=(
                    f"{judgement.reason} "
                    f"(judge confidence={judgement.confidence:.2f})"
                ),
            )

        except Exception as e:
            logger.exception("LLM Judge evaluation failed")

            return EvalResult(
                eval_name="evaluate_intent_clarity",
                eval_type=EvalType.LLM_JUDGE,
                passed=False,
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
            )

    @staticmethod
    def _parse_judgement(result: dict[str, Any]) -> RubricJudgeResult:
        """
        Parse and validate the judge's structured output.

        Adjust this depending on whether your LLM client returns
        a dict or a JSON string.
        """

        response = result.get("response")

        if response is None:
            raise ValueError("Judge returned no response")

        if isinstance(response, str):
            import json
            response = json.loads(response)

        return RubricJudgeResult.model_validate(response)
