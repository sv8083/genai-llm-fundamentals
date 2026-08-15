from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_analyze_ticket_returns_structured_response(monkeypatch):
    class FakeLLMClient:
        def generate(self, prompt: str):
            return {
                "category": "billing",
                "intent": "refund",
                "priority": "high",
                "order_id": "12345",
                "sentiment": "negative",
                "summary": "Customer reports a duplicate payment and requests a refund.",
            }

    monkeypatch.setattr("support_ticket.api.get_llm_client", lambda: FakeLLMClient())

    response = client.post(
        "/ticket/analyze",
        json={"message": "My payment was deducted twice for order 12345."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "billing"
    assert payload["intent"] == "refund"
    assert payload["priority"] == "high"
    assert payload["order_id"] == "12345"
    assert payload["sentiment"] == "negative"
    assert "duplicate payment" in payload["summary"].lower()
