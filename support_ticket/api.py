import json

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from prompts.load import PromptLoader
from llm import get_llm_client
from support_ticket.schema import TicketAnalysis, TicketAnalyzeRequest
from telemetry import PhoenixTelemetry
from core.settings import settings

router = APIRouter()


@router.post("/analyze", response_model=TicketAnalysis)
async def analyze_ticket(data: TicketAnalyzeRequest):
    prompt = PromptLoader.render(
        "prod/v1",
        message=data.message,
    )

    # Extract model config from request
    model_config = data.model or {}
    model_name = model_config.name if model_config and model_config.name else settings.ollama_model
    temperature = model_config.temperature if model_config and model_config.temperature is not None else settings.ollama_temperature
    top_p = model_config.top_p if model_config and model_config.top_p is not None else settings.ollama_top_p
    
    # Create LLM client with optional model name override
    client = get_llm_client(base_url=None, model=model_name)
    result = None
    
    try:
        # Trace LLM operation with telemetry
        with PhoenixTelemetry.trace_llm_call(
            "ticket_analysis",
            attributes={
                "message_length": len(data.message),
                "model": client.model,
                "temperature": temperature or client.temperature,
                "top_p": top_p or client.top_p,
            }
        ) as span:
            # Pass optional temperature and top_p parameters
            llm_data = client.generate(
                prompt,
                temperature=temperature,
                top_p=top_p,
            )
            if span:
                span.set_attribute("response_received", True)

            if not llm_data.get("response"):
                raise HTTPException(status_code=400, detail="Unable to fetch data")
            
            result = TicketAnalysis.model_validate(llm_data["response"])
            if span:
                span.set_attribute("category", result.category)
                span.set_attribute("sentiment", result.sentiment)
                span.set_attribute("priority", result.priority)
            
            return result
            
    except HTTPException:
        raise
    except ValidationError as _val_err:
        raise HTTPException(status_code=400, detail=f"Error in forming response {_val_err}")
    except Exception as exc:  # pragma: no cover - network failure path
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
