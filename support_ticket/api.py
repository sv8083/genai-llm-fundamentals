import json

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from support_ticket.llm import get_llm_client
from support_ticket.schema import TicketAnalysis, TicketAnalyzeRequest

router = APIRouter()


@router.post("/analyze", response_model=TicketAnalysis)
async def analyze_ticket(data: TicketAnalyzeRequest):
    prompt = f"""
**Role**
You are an expert support ticket classification assistant. Your job is to analyze customer support messages and extract structured information to route and prioritize tickets effectively.

**Task**
Analyze the following customer message and classify it according to predefined categories, intent, priority, and sentiment.

**Categories**
- Billing/Payment
- Technical Support
- Account Management
- Product Inquiry
- Complaint
- Feature Request
- Other

**Rules**
- Always extract the order_id if present in the message
- Provide accurate sentiment analysis (positive, negative, neutral)
- Priority levels: critical, high, medium, low
- Intent should be concise and actionable
- If a value cannot be determined, indicate as null

**Output Requirements**
Return valid JSON only with these keys and definitions:
- category: The ticket category (e.g., Billing/Payment, Technical Support, Account Management, Product Inquiry, Complaint, Feature Request, Other)
- intent: A concise, actionable description of what the customer wants or needs
- priority: Priority level - 'critical' for urgent issues, 'high' for important problems, 'medium' for standard issues, 'low' for non-urgent requests
- order_id: The order ID if mentioned in the message, otherwise null
- sentiment: Sentiment analysis result - 'positive' for satisfied customers, 'neutral' for factual inquiries, 'negative' for unhappy or frustrated customers
- summary: A brief summary of the ticket content and key issue

**Examples**

Example 1:
Input: "My order #12345 hasn't arrived yet. It's been 2 weeks and I'm very frustrated!"
Output:
{{
  "category": "Complaint",
  "intent": "Track order delivery status",
  "priority": "high",
  "order_id": "12345",
  "sentiment": "negative",
  "summary": "Customer complaining about delayed delivery for order 2 weeks overdue"
}}

Example 2:
Input: "Does your product work with Windows 11?"
Output:
{{
  "category": "Product Inquiry",
  "intent": "Verify product compatibility",
  "priority": "medium",
  "order_id": null,
  "sentiment": "neutral",
  "summary": "Customer inquiring about Windows 11 compatibility"
}}

Example 3:
Input: "I'd love to see a dark mode option in your app!"
Output:
{{
  "category": "Feature Request",
  "intent": "Request dark mode feature",
  "priority": "low",
  "order_id": null,
  "sentiment": "positive",
  "summary": "Customer requesting dark mode feature for the application"
}}

**Customer Message**
{data.message}
"""

    client = get_llm_client()

    try:
        llm_data = client.generate(prompt)
    except Exception as exc:  # pragma: no cover - network failure path
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    if not llm_data.get("response"):
        raise HTTPException(status_code=400, detail="Unable to fetch data")
    try:
        return TicketAnalysis.model_validate(llm_data["response"])
    except ValidationError as _val_err:
        raise HTTPException(status_code=400, detail=f"Error in forming response {_val_err}")
