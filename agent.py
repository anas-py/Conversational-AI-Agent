"""
AutoStream Conversational AI Agent
===================================
Built with LangGraph for state management, RAG for knowledge retrieval,
and tool execution for lead capture.

Author: ML Intern Assignment — ServiceHive / Inflx
"""

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ──────────────────────────────────────────────
# 1. KNOWLEDGE BASE LOADER (RAG)
# ──────────────────────────────────────────────

KB_PATH = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def build_rag_context(kb: dict) -> str:
    """Convert the JSON knowledge base into a clean text block for the LLM context."""
    plans = kb["pricing"]["plans"]
    policies = kb["policies"]
    faqs = kb["faqs"]
    company = kb["company"]

    plan_text = ""
    for plan in plans:
        features_str = "\n    - ".join(plan["features"])
        plan_text += (
            f"\n  [{plan['name']} Plan — ${plan['price_monthly']}/month]\n"
            f"    - {features_str}\n"
            f"  Best for: {plan['best_for']}\n"
        )

    faq_text = ""
    for faq in faqs:
        faq_text += f"\n  Q: {faq['question']}\n  A: {faq['answer']}\n"

    return f"""
=== AutoStream Knowledge Base ===

Company: {company['name']}
Description: {company['description']}

--- PRICING ---
{plan_text}
--- POLICIES ---
- Refund Policy: {policies['refund_policy']}
- Free Trial: {policies['free_trial']}
- Billing: {policies['billing']}
- Basic Plan Support: {policies['support']['basic_plan']}
- Pro Plan Support: {policies['support']['pro_plan']}

--- FAQs ---
{faq_text}
"""


# ──────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ──────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API call to simulate saving a lead to the CRM.
    In production, this would POST to a real endpoint.
    """
    print(f"\n{'='*50}")
    print(f"✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ──────────────────────────────────────────────
# 3. LANGGRAPH STATE DEFINITION
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (LangGraph auto-appends with add_messages)
    messages: Annotated[list[BaseMessage], add_messages]

    # Intent classification for the latest user turn
    intent: Literal["greeting", "product_inquiry", "high_intent", "unknown"]

    # Lead info collected progressively — None until provided by user
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None

    # Whether we've already triggered the lead capture tool
    lead_captured: bool

    # Which field we're currently waiting for from the user
    collecting_field: Literal["name", "email", "platform", "none"]


# ──────────────────────────────────────────────
# 4. LLM SETUP
# ──────────────────────────────────────────────

KB = load_knowledge_base()
RAG_CONTEXT = build_rag_context(KB)

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)


# ──────────────────────────────────────────────
# 5. INTENT DETECTION NODE
# ──────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """
You are an intent classifier for AutoStream, a SaaS video editing platform.

Classify the user's latest message into EXACTLY one of these intents:
- greeting        → casual hello, hi, hey, how are you, etc.
- product_inquiry → asking about features, pricing, plans, support, refunds, trial
- high_intent     → ready to sign up, wants to try, wants to buy, wants to subscribe, asks to get started
- unknown         → anything else

Respond with ONLY the intent label. No explanation. No punctuation.
""".strip()


def detect_intent_node(state: AgentState) -> dict:
    """Classify the intent of the latest human message."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"intent": "unknown"}

    result = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human.content),
    ])

    intent_raw = result.content.strip().lower()
    valid_intents = {"greeting", "product_inquiry", "high_intent", "unknown"}
    intent = intent_raw if intent_raw in valid_intents else "unknown"

    return {"intent": intent}


# ──────────────────────────────────────────────
# 6. RESPONSE GENERATION NODE
# ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = f"""
You are Maya, a friendly and knowledgeable sales assistant for AutoStream — a SaaS platform
that provides AI-powered automated video editing tools for content creators.

Your personality:
- Warm, conversational, and helpful (never robotic)
- Concise — don't overwhelm users with walls of text
- Honest — never make up features or prices not in the knowledge base

Use the following verified knowledge base to answer any product or pricing questions:

{RAG_CONTEXT}

Important rules:
1. Only use information from the knowledge base above — never invent features or prices.
2. When a user shows high intent (wants to sign up / try the product), acknowledge it warmly
   and tell them you'll collect a few quick details to get them started.
3. When collecting lead info, ask for ONE field at a time in this order: name → email → platform.
4. Once all three are collected, confirm warmly and let them know the team will reach out.
5. Keep responses human and natural. Avoid bullet-heavy responses for casual exchanges.
""".strip()


def generate_response_node(state: AgentState) -> dict:
    """Generate the agent's reply based on current state."""

    # If we're mid-collection, we handle that in the lead node — skip here
    if state.get("collecting_field", "none") != "none":
        return {}

    system = SystemMessage(content=AGENT_SYSTEM_PROMPT)
    response = llm.invoke([system] + state["messages"])

    return {"messages": [AIMessage(content=response.content)]}


# ──────────────────────────────────────────────
# 7. LEAD COLLECTION NODE
# ──────────────────────────────────────────────

def extract_email(text: str) -> str | None:
    match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


def lead_collection_node(state: AgentState) -> dict:
    """
    Progressively collect name, email, and platform from the user.
    Triggers mock_lead_capture() only when all three are collected.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    user_text = last_human.content.strip() if last_human else ""

    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    collecting = state.get("collecting_field", "none")

    # ── Fill in the field we were waiting for ──
    if collecting == "name":
        name = user_text
    elif collecting == "email":
        email = extract_email(user_text) or user_text
    elif collecting == "platform":
        platform = user_text

    # ── Decide what to ask next ──
    if not name:
        reply = "Awesome! I'd love to get you set up. Could you share your name first? 😊"
        next_field = "name"

    elif not email:
        reply = f"Great to meet you, {name}! What's the best email address to reach you at?"
        next_field = "email"

    elif not platform:
        reply = "Almost there! Which platform are you primarily creating content on? (e.g., YouTube, Instagram, TikTok)"
        next_field = "platform"

    else:
        # All three collected — fire the tool
        result = mock_lead_capture(name, email, platform)
        reply = (
            f"You're all set, {name}! 🎉\n\n"
            f"We've got your details and our team will be in touch at **{email}** shortly "
            f"to get your AutoStream Pro account activated for your {platform} channel.\n\n"
            f"In the meantime, feel free to start your **7-day free trial** — no credit card needed. "
            f"Is there anything else I can help you with?"
        )
        next_field = "none"
        return {
            "messages": [AIMessage(content=reply)],
            "lead_name": name,
            "lead_email": email,
            "lead_platform": platform,
            "lead_captured": True,
            "collecting_field": "none",
        }

    return {
        "messages": [AIMessage(content=reply)],
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "collecting_field": next_field,
    }


# ──────────────────────────────────────────────
# 8. ROUTING LOGIC
# ──────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    Route to the appropriate node based on intent and current collection state.
    """
    # If we're already mid-collection, keep collecting regardless of intent
    if state.get("collecting_field", "none") != "none":
        return "lead_collection"

    # If lead was already captured, just respond normally
    if state.get("lead_captured"):
        return "generate_response"

    intent = state.get("intent", "unknown")
    if intent == "high_intent":
        return "lead_collection"
    else:
        return "generate_response"


# ──────────────────────────────────────────────
# 9. BUILD THE LANGGRAPH
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("lead_collection", lead_collection_node)

    # Entry point
    graph.add_edge(START, "detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "generate_response": "generate_response",
            "lead_collection": "lead_collection",
        },
    )

    # Both response nodes terminate the turn
    graph.add_edge("generate_response", END)
    graph.add_edge("lead_collection", END)

    return graph.compile()


# ──────────────────────────────────────────────
# 10. MAIN CHAT LOOP
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  AutoStream AI Assistant  |  Powered by Inflx (ServiceHive)")
    print("="*60)
    print("  Type 'quit' or 'exit' to end the conversation.\n")

    agent = build_graph()

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": "unknown",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_field": "none",
    }

    # Opening message from the agent
    opening = (
        "Hey there! 👋 Welcome to AutoStream — the easiest way to automate your video editing.\n"
        "I'm Maya, your assistant. Whether you have questions about pricing, features, or want "
        "to get started, I'm here to help! What's on your mind?"
    )
    print(f"Maya: {opening}\n")
    state["messages"].append(AIMessage(content=opening))

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nMaya: Thanks for chatting! Have an amazing day. 🚀\n")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        result = agent.invoke(state)

        # Merge result back into state (LangGraph returns full state)
        state.update(result)

        # Print the latest AI message
        last_ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if last_ai:
            print(f"\nMaya: {last_ai.content}\n")


if __name__ == "__main__":
    main()
