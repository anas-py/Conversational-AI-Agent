# AutoStream AI Sales Agent

> **Assignment Project** — ServiceHive / Inflx ML Internship  
> A production-style conversational AI agent built with LangGraph, RAG, and Claude Haiku.

---

## What This Is

AutoStream is a fictional SaaS platform for automated video editing. This repo contains a fully functional AI sales agent ("Maya") that can:

- Answer pricing and feature questions from a local knowledge base (RAG)
- Detect user intent in real time (greeting / product inquiry / high-intent)
- Progressively collect lead info (name, email, platform) without being pushy
- Fire a mock lead-capture API function only after all fields are collected
- Retain full conversation history across 5–6+ turns using LangGraph state

---

## How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
# Open .env and paste your Anthropic API key
```

Your `.env` should look like:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run the agent

```bash
python agent.py
```

You'll see Maya greet you and the conversation begins in your terminal.

---

## Example Conversation

```
Maya: Hey there! 👋 Welcome to AutoStream...

You: Hi, what are your pricing plans?

Maya: Great question! We have two plans:
  - Basic ($29/month): 10 videos/month, 720p export, email support
  - Pro ($79/month): Unlimited videos, 4K, AI captions, 24/7 support

You: The Pro plan sounds good. I want to sign up for my YouTube channel.

Maya: Awesome! I'd love to get you set up. Could you share your name first? 😊

You: My name is Arjun

Maya: Great to meet you, Arjun! What's the best email address to reach you at?

You: arjun@gmail.com

Maya: Almost there! Which platform are you primarily creating content on?

You: YouTube

✅  LEAD CAPTURED: Arjun, arjun@gmail.com, YouTube

Maya: You're all set, Arjun! 🎉 Our team will be in touch shortly...
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it treats conversation state as a **typed, explicit data structure** (a `TypedDict`) rather than relying on implicit message threading. This is critical for a lead-capture workflow — we need to track `lead_name`, `lead_email`, `lead_platform`, and `collecting_field` as first-class state, not parse them back out of conversation history.

LangGraph's `StateGraph` also makes **conditional routing** clean and readable. The `route_after_intent` function can look at the full state to decide whether to generate a normal response or continue collecting lead fields — something that would require messy message parsing in AutoGen.

### How State Is Managed

Every conversation turn flows through three nodes:

```
[User Input] → detect_intent → (route) → generate_response OR lead_collection → [END]
```

1. **detect_intent** — A fast LLM call classifies intent into `greeting`, `product_inquiry`, `high_intent`, or `unknown`.
2. **generate_response** — Uses the full message history + a RAG-injected system prompt to reply accurately.
3. **lead_collection** — Reads the current `collecting_field` from state, fills it with the latest user message, and asks for the next field. Only calls `mock_lead_capture()` when all three fields are populated.

LangGraph's `add_messages` reducer automatically appends new messages to history without overwriting it, giving the agent complete memory across all turns.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business Cloud API** (Meta) with an **outbound webhook**. Here's the integration flow:

### Architecture

```
WhatsApp User
     │
     ▼
Meta WhatsApp Cloud API
     │  (POST /webhook — incoming message JSON)
     ▼
Your FastAPI / Flask Server
     │
     ├── Retrieve session state from Redis (keyed by phone number)
     │
     ├── Run agent.invoke(state) with new HumanMessage
     │
     ├── Save updated state back to Redis
     │
     └── POST reply to WhatsApp Send Message API
           └── User receives Maya's reply on WhatsApp
```

### Key Steps

1. **Register a webhook** on Meta Developer Console pointing to your server's `/webhook` endpoint.
2. **Verify the webhook** by responding to Meta's `GET` challenge with the `hub.challenge` token.
3. **On each POST**, extract `messages[0].text.body` and the sender's `wa_id` (phone number).
4. **Load session state** from Redis using `wa_id` as the key — this replaces in-memory state.
5. **Invoke the LangGraph agent** and get the reply.
6. **Save updated state** to Redis with a TTL (e.g., 30 minutes of inactivity resets the session).
7. **POST the reply** to `https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages`.

### Sample Webhook Handler (FastAPI)

```python
from fastapi import FastAPI, Request
import redis, json
from agent import build_graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()
r = redis.Redis()
agent = build_graph()

@app.post("/webhook")
async def handle_message(request: Request):
    body = await request.json()
    msg = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone = msg["from"]
    text = msg["text"]["body"]

    # Load or init state
    raw = r.get(f"session:{phone}")
    state = json.loads(raw) if raw else {
        "messages": [], "intent": "unknown",
        "lead_name": None, "lead_email": None,
        "lead_platform": None, "lead_captured": False,
        "collecting_field": "none"
    }

    state["messages"].append(HumanMessage(content=text))
    result = agent.invoke(state)
    state.update(result)
    r.setex(f"session:{phone}", 1800, json.dumps(state))  # 30min TTL

    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    # TODO: POST last_ai.content to WhatsApp Send API
    return {"status": "ok"}
```

This approach keeps the LangGraph agent completely stateless on the server side — all conversation state lives in Redis, keyed by phone number.

---

## Project Structure

```
autostream-agent/
├── agent.py                    # Main agent (LangGraph + RAG + tool)
├── knowledge_base/
│   └── autostream_kb.json      # Local knowledge base for RAG
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Evaluation Checklist

| Criterion | Implementation |
|---|---|
| Intent detection | `detect_intent_node` — LLM classifies every turn |
| RAG knowledge retrieval | `build_rag_context()` injects KB into system prompt |
| State management | LangGraph `AgentState` TypedDict with `add_messages` |
| Progressive lead collection | `lead_collection_node` — one field at a time |
| Tool only fires when complete | Guard condition checks all 3 fields before calling `mock_lead_capture` |
| 5–6 turn memory | LangGraph message history — no truncation |
| WhatsApp deployment | Documented above with Redis session management |

---

## Tech Stack

- **Python 3.9+**
- **LangGraph** — state machine and graph orchestration
- **LangChain** — message types, LLM abstraction
- **Claude Haiku (claude-haiku-4-5)** — fast, cost-efficient LLM
- **JSON knowledge base** — lightweight local RAG
