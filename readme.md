## Mini Project — Local LLM Support Ticket Intelligence

Build a small production-style API that takes a customer support message and turns it into a **validated structured support ticket**.

### What the system does

```text
User message
     ↓
Prompt construction
     ↓
Local LLM
     ↓
Structured response
     ↓
Pydantic validation
     ↓
Business validation
     ↓
Final API response
```

Example input

```text
My payment was deducted twice for order 12345.
Please refund the extra amount.
```

Expected output

```json
{
  "category": "billing",
  "intent": "refund",
  "priority": "high",
  "order_id": "12345",
  "sentiment": "negative",
  "summary": "Customer reports a duplicate payment and requests a refund."
}
```

---

# What concepts this project covers

| Concept            | How you will use it                                   |
| ------------------ | ----------------------------------------------------- |
| Tokens             | Understand input and output token usage               |
| Context            | Construct the model input                             |
| Transformer        | Understand what happens inside the local model        |
| Attention          | Understand model behavior conceptually                |
| Inference          | Run the local model                                   |
| Logits             | Understand generation internally                      |
| Temperature        | Experiment with deterministic versus varied responses |
| Top P              | Experiment with sampling                              |
| Structured output  | Force a predictable response                          |
| Pydantic           | Validate model output                                 |
| Prompt engineering | Design the classification prompt                      |
| Few shot prompting | Give examples                                         |
| Prompt versioning  | Maintain prompt versions                              |
| Model selection    | Compare two local models                              |
| Fine tuning        | Understand why you are not using it here              |
| Hallucination      | Deliberately test unsupported claims                  |
| Failure handling   | Handle malformed or invalid responses                 |
| Model limitations  | Create adversarial and ambiguous inputs               |

---

# Architecture

Keep the architecture intentionally simple.

```text
                    FastAPI
                       ↓
                Request validation
                       ↓
                Prompt Builder
                       ↓
                 Local LLM
                       ↓
              Structured response
                       ↓
                 Pydantic
                       ↓
             Business validation
                       ↓
                    JSON
```

---

# Suggested tech stack

```text
Python
FastAPI
Pydantic
Ollama
httpx
pytest
```

---

# Project structure

```text
llm-ticket-intelligence/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── tickets.py
│   │
│   ├── llm/
│   │   ├── client.py
│   │   ├── prompts.py
│   │   └── config.py
│   │
│   ├── schemas/
│   │   └── ticket.py
│   │
│   ├── services/
│   │   └── ticket_service.py
│   │
│   └── validators/
│       └── ticket_validator.py
│
├── prompts/
│   ├── v1.txt
│   └── v2.txt
│
├── tests/
│   ├── test_api.py
│   ├── test_validation.py
│   └── test_llm_cases.py
│
├── .env
├── requirements.txt
└── README.md
```

---

# Phase 1 — Basic LLM call

Start extremely simple.

```text
POST /tickets/analyze
```

Request

```json
{
  "message": "My payment was deducted twice for order 12345."
}
```

Your application calls the local model.

Initially allow free text.

The goal here is simply to understand

```text
FastAPI
   ↓
Prompt
   ↓
Ollama
   ↓
Response
```

---

# Phase 2 — Prompt engineering

Now improve the prompt.

Your prompt should define

```text
Role
Task
Categories
Rules
Output requirements
```

For example

```text
You are a customer support classification assistant.

Classify the customer message.

Supported categories are

billing
account
technical
refund

Do not invent categories.

Extract the order identifier when present.

Return the required structured response.
```

Then add examples.

```text
Example

Message
I was charged twice.

Category
billing


Example

Message
I want my money back.

Category
refund
```

Now compare zero shot and few shot performance.

---

# Phase 3 — Structured output

Now make the model return a defined structure.

Create

```python
class TicketAnalysis(BaseModel):
    category: Literal[
        "billing",
        "account",
        "technical",
        "refund"
    ]

    intent: str
    priority: Literal["low", "medium", "high"]
    order_id: str | None
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str
```

Then validate the LLM response.

```text
LLM
 ↓
dict
 ↓
TicketAnalysis.model_validate
 ↓
Valid
```

or

```text
LLM
 ↓
dict
 ↓
Pydantic
 ↓
ValidationError
```

---

# Phase 4 — Business validation

Now make it more realistic.

Suppose the model returns

```json
{
  "category": "refund",
  "priority": "high",
  "order_id": null
}
```

The schema might accept it.

But your business rule could say

```text
Refund requests require an order ID.
```

Therefore

```text
Schema validation
        ↓
Business validation
        ↓
Accept / Reject
```

This is an important distinction from our previous topic.

---

# Phase 5 — Temperature experiment

Now run the **same input multiple times**.

Test something like

```text
temperature = 0
temperature = 0.3
temperature = 0.7
temperature = 1.0
```

Record

```text
Input
Temperature
Output
Latency
Tokens
```

You should observe how generation behavior changes.

Then answer

> Does temperature zero guarantee identical output with my local setup?

This becomes a practical experiment rather than something you merely memorized.

---

# Phase 6 — Top P experiment

Repeat the experiment with different Top P values.

For example

```text
Top P = 0.5
Top P = 0.8
Top P = 0.95
1.0
```

Keep the other parameters controlled as much as possible.

Your goal is to understand experimentally

```text
Temperature
      +
Top P
      ↓
Generation behavior
```

---

# Phase 7 — Model comparison

Now run the exact same evaluation dataset against two local models.

For example

```text
Model A
Model B
```

Don't select based on which response "feels better."

Create a small test set.

```text
20 normal cases
10 ambiguous cases
10 edge cases
10 malformed cases
10 adversarial cases
```

Then measure

```text
Schema success rate
Classification accuracy
Invalid output rate
Average latency
Average output tokens
```

Now you can make an actual model selection decision.

---

# Phase 8 — Hallucination experiment

Create inputs where the required information is missing.

For example

```text
I want a refund for my order.
```

There is no order ID.

See whether the model

```text
Correctly returns null
```

or

```text
Invents an order ID
```

That gives you a practical demonstration of hallucination.

Then improve the prompt

```text
Never invent an order ID.

If an order ID is not present in the customer message,
return null.
```

Run the same evaluation again.

This teaches you an important lesson

> Prompting can reduce some failure modes, but it does not guarantee correctness.

---

# Phase 9 — Malformed output handling

Now deliberately make the system robust.

Your pipeline becomes

```text
LLM
 ↓
Parse
 ↓
Pydantic
 ↓
Success
```

or

```text
LLM
 ↓
Parse
 ↓
Pydantic
 ↓
Failure
 ↓
Bounded retry
 ↓
Pydantic
 ↓
Success
```

If it still fails

```text
Failure
 ↓
Graceful error
```

Track how often this happens.

---

# Phase 10 — Prompt versioning

Create

```text
prompts/
    v1.txt
    v2.txt
```

Run your evaluation dataset against both.

For example

```text
                 V1       V2

Accuracy         82%      91%
Schema success   88%      97%
Latency          1.8s     1.9s
Tokens           210      225
```

Now you can explain in an interview

> We don't change production prompts blindly. We evaluate prompt versions against a fixed dataset and promote them based on measured performance.

That is a **very good Lead Engineer statement**.

---

# The final mini project

By the end, your system should look like this

```text
                       Client
                         ↓
                     FastAPI
                         ↓
                 Request validation
                         ↓
                  Prompt version
                         ↓
                  Prompt builder
                         ↓
                   Local LLM
                         ↓
                Structured response
                         ↓
                 JSON parsing
                         ↓
                 Pydantic validation
                         ↓
                Business validation
                         ↓
               Retry if recoverable
                         ↓
                    Final result
```

And alongside it

```text
                    Evaluation
                        ↓
          +-------------+-------------+
          ↓             ↓             ↓
       Model A       Model B       Prompt V2
          ↓             ↓             ↓
          +-------------+-------------+
                        ↓
                   Comparison
```

---

# What I specifically DON'T want you to add yet

Keep these **out of this project**.

```text
❌ RAG
❌ Vector database
❌ Embeddings for retrieval
❌ Agents
❌ MCP
❌ Tool calling
❌ Long term memory
❌ Multi agent
❌ Guardrails
❌ AI security
❌ Governance
❌ Complex observability
❌ Fine tuning
```

We'll build those into later projects.

That way, each project corresponds to a section of your interview preparation rather than creating one giant project where you don't understand which component is doing what.

---

# Why I like this project for you

The most valuable part isn't the API.

It's that you'll be able to **demonstrate the concepts we just studied experimentally**.

By the end, you should be able to explain

```text
Why did this model produce this output

What happens when temperature changes

What happens when Top P changes

Why did the model hallucinate

Why did Pydantic reject the output

Why did the retry happen

Why is Model A better than Model B

Why is Prompt V2 better than Prompt V1

Why can't we trust the LLM output directly
```

Those are exactly the kinds of questions that turn theoretical interview preparation into **actual engineering understanding**.

### Recommended next step

Don't jump into coding the entire project at once.

Build it in **five small milestones**

```text
M1
FastAPI → Local LLM

M2
Prompt engineering → Structured output

M3
Pydantic → Business validation

M4
Temperature → Top P → Model comparison

M5
Evaluation dataset → Retry → Prompt versioning
```

Once you've built M1 through M5, you'll have a small but genuinely useful **LLM application engineering project** that covers the entire LLM Fundamentals section without prematurely mixing in RAG or agent architecture.
