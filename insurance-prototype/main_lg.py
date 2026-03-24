"""
Virtual Insurance Agency — LangGraph version

Same 4-stage pipeline as main.py, rewritten as a LangGraph graph.

Graph (linear — no branching needed):
  [receptionist] → [classifier] → [specialist] → [underwriter] → END

DRY/KISS changes vs main.py:
  - State TypedDict replaces 4 local variables + manual passing between stages
  - _converse() / _call_once() are shared helpers used by all four nodes
  - main() is just graph.invoke() — pipeline logic lives in the graph topology
  - specialist prompt selection is data-driven (state["product"]) not structural
"""

import json
import typing
import uuid
from pathlib import Path
from anthropic import Anthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

client = Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_TURNS = 15

# ---------------------------------------------------------------------------
# Typed models — defined first so prompts can derive their JSON examples
# ---------------------------------------------------------------------------

class ClientProfile(BaseModel):
    business_name: str = ""
    owner_name: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    state: str = ""
    business_type: str = ""
    employees: int = 0
    revenue: str = ""
    years_in_business: int = 0


def _int_field(annotation) -> bool:
    """True for bare int or any Optional[int] / int | None annotation."""
    if annotation is int:
        return True
    return int in typing.get_args(annotation) and type(None) in typing.get_args(annotation)


def _done_json_example(model_cls) -> str:
    """Build the terminal JSON example Claude must emit, derived from the model fields.
    Int fields use 0 (valid JSON int); all others use '...'."""
    placeholders = {
        name: (0 if _int_field(f.annotation) else "...")
        for name, f in model_cls.model_fields.items()
    }
    return json.dumps({"done": True, "data": placeholders})


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RECEPTIONIST_PROMPT = f"""You are a professional insurance agency receptionist.
Collect the following from the client through natural conversation:
  1. Business name
  2. Owner / contact name
  3. Email address
  4. Phone number
  5. City and state
  6. What the business does (2-3 sentences)
  7. Number of employees
  8. Approximate annual revenue
  9. Years in business

Ask 1-2 questions at a time. Be friendly but concise.

When you have ALL nine items, output ONLY this JSON and nothing else:
{_done_json_example(ClientProfile)}
"""

CLASSIFIER_PROMPT = """You are an insurance product classifier.
Given a business profile, choose the single best insurance product.

Products:
  general_liability      — physical premises, customers visit, property risk
  professional_liability — advice / consulting / services, errors & omissions
  commercial_auto        — vehicles used for business operations
  workers_comp           — employees doing physical or hazardous work

Respond ONLY with JSON, no other text:
{"product": "<key>", "reason": "<one sentence>"}
"""

SPECIALIST_PROMPTS = {
    "general_liability": """You are a General Liability underwriting specialist.
Collect these details in natural conversation:
  1. Do clients visit your premises?
  2. Do you handle third-party property?
  3. Prior claims in the last 3 years? (number)
  4. Primary business state?
  5. Desired coverage limit — $1M, $2M, or $5M?

When done, output ONLY this JSON and nothing else:
{"done": true, "data": {"clients_on_premises": <bool>, "handles_property": <bool>, "prior_claims": <int>, "state": "...", "coverage_limit": "..."}}
""",
    "professional_liability": """You are a Professional Liability (E&O) underwriting specialist.
Collect in natural conversation:
  1. Specific professional services provided?
  2. Largest single contract value?
  3. Do you use written contracts with all clients?
  4. Prior E&O claims in last 3 years? (number)
  5. Desired coverage limit — $500K, $1M, or $2M?

When done, output ONLY this JSON and nothing else:
{"done": true, "data": {"services": "...", "largest_contract": "...", "written_contracts": <bool>, "prior_claims": <int>, "coverage_limit": "..."}}
""",
    "commercial_auto": """You are a Commercial Auto underwriting specialist.
Collect in natural conversation:
  1. Number of business vehicles?
  2. Vehicle types (sedans, trucks, vans, etc.)?
  3. Primary use (delivery, sales, transport)?
  4. Any drivers under 25?
  5. Accidents or violations in the last 3 years? (number)

When done, output ONLY this JSON and nothing else:
{"done": true, "data": {"vehicle_count": <int>, "vehicle_types": "...", "primary_use": "...", "young_drivers": <bool>, "incidents": <int>}}
""",
    "workers_comp": """You are a Workers Compensation underwriting specialist.
Collect in natural conversation:
  1. Job classifications (office, field, construction, etc.)?
  2. States where employees work?
  3. Workplace injuries in last 3 years? (number)
  4. Do you use subcontractors?
  5. Total annual payroll?

When done, output ONLY this JSON and nothing else:
{"done": true, "data": {"job_classes": "...", "states": "...", "prior_injuries": <int>, "uses_subs": <bool>, "payroll": "..."}}
""",
}

UNDERWRITER_PROMPT = """You are a senior insurance underwriter.
Given a client profile and underwriting details, produce a realistic quote.

Respond ONLY with this JSON structure and nothing else:
{
  "quote_id": "Q-<6 random digits>",
  "client": { <copy client fields> },
  "product": "<product key>",
  "coverage_limit": "<limit string>",
  "annual_premium": <integer>,
  "monthly_premium": <integer>,
  "deductible": <integer>,
  "exclusions": ["<item>", "<item>"],
  "notes": "<brief underwriter note>",
  "valid_days": 30
}
"""

# ---------------------------------------------------------------------------
# State — single source of truth flowing through all nodes
# ---------------------------------------------------------------------------

class State(BaseModel):
    client_data: ClientProfile = ClientProfile()
    product: str = ""
    underwriting_data: dict = {}
    quote: dict = {}

# ---------------------------------------------------------------------------
# Shared helpers — each used by exactly two nodes, no duplication
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first balanced JSON object from a string using brace counting.
    Avoids the greedy-regex trap where trailing prose or multi-object responses
    produce an unparseable span."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _converse(system_prompt: str, label: str) -> dict:
    """
    Conversational loop shared by receptionist and specialist nodes.
    Runs until agent emits terminal JSON {"done": true, "data": {...}}.
    Returns the parsed 'data' dict.
    """
    print(f"\n{'─'*54}\n  {label}\n{'─'*54}\n")
    messages = [{"role": "user", "content": "Please begin."}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
        reply = response.content[0].text.strip()
        messages.append({"role": "assistant", "content": reply})

        parsed = _extract_json(reply)
        if parsed and parsed.get("done"):
            print(f"[{label}]: Thank you, I have all the information I need.\n")
            return parsed.get("data", {})

        print(f"[{label}]: {reply}\n")
        user_input = input("You: ").strip() or "Please continue."
        messages.append({"role": "user", "content": user_input})

    raise RuntimeError(f"{label}: max turns ({MAX_TURNS}) reached without completing intake — aborting pipeline.")


def _call_once(system_prompt: str, user_message: str) -> dict:
    """
    One-shot internal agent shared by classifier and underwriter nodes.
    No user interaction — returns parsed JSON or raises on failure.
    """
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}") from e
    if not response.content:
        raise RuntimeError("Empty response from API")
    parsed = _extract_json(response.content[0].text)
    if parsed is None:
        raise RuntimeError(f"No JSON in response: {response.content[0].text[:200]}")
    return parsed

# ---------------------------------------------------------------------------
# Nodes — each reads from state, returns only what it changes
# ---------------------------------------------------------------------------

def receptionist_node(state: State) -> dict:
    data = _converse(RECEPTIONIST_PROMPT, "Receptionist")
    # Validate and coerce raw dict into typed model (Pydantic ignores unknown keys)
    profile = ClientProfile(**data) if data else ClientProfile()
    print(f"[System]: Profile collected → {profile.model_dump()}\n")
    return {"client_data": profile}


def classifier_node(state: State) -> dict:
    result = _call_once(
        CLASSIFIER_PROMPT,
        f"Classify this business for insurance: {json.dumps(state.client_data.model_dump())}",
    )
    product = result.get("product", "")
    if product not in SPECIALIST_PROMPTS:
        print(f"[System]: WARNING — unknown product '{product}', defaulting to general_liability.")
        product = "general_liability"
    print(f"[System]: Routing to {product.replace('_', ' ').title()} specialist.")
    print(f"[Reason]: {result.get('reason', '')}\n")
    return {"product": product}


def specialist_node(state: State) -> dict:
    # Data-driven prompt selection — no conditional edges needed
    prompt = SPECIALIST_PROMPTS.get(state.product, SPECIALIST_PROMPTS["general_liability"])
    label = f"{state.product.replace('_', ' ').title()} Specialist"
    data = _converse(prompt, label)
    print(f"[System]: Underwriting data collected → {json.dumps(data)}\n")
    return {"underwriting_data": data}


def underwriter_node(state: State) -> dict:
    print("[System]: Generating quote...\n")
    quote = _call_once(
        UNDERWRITER_PROMPT,
        json.dumps({
            "client": state.client_data.model_dump(),
            "product": state.product,
            "underwriting_details": state.underwriting_data,
        }),
    )
    print("=" * 54)
    print("  INSURANCE QUOTE")
    print("=" * 54)
    print(json.dumps(quote, indent=2))
    print("=" * 54)

    qid = quote.get("quote_id", "draft").replace("/", "-")
    out_path = Path(__file__).parent / f"quote_{qid}.json"
    with open(out_path, "w") as f:
        json.dump(quote, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return {"quote": quote}

# ---------------------------------------------------------------------------
# Entry point — graph built here so checkpointer opens at run time, not import
# ---------------------------------------------------------------------------

def main():
    builder = StateGraph(State)

    builder.add_node("receptionist", receptionist_node)
    builder.add_node("classifier",   classifier_node)
    builder.add_node("specialist",   specialist_node)
    builder.add_node("underwriter",  underwriter_node)

    builder.set_entry_point("receptionist")
    builder.add_edge("receptionist", "classifier")
    builder.add_edge("classifier",   "specialist")
    builder.add_edge("specialist",   "underwriter")
    builder.add_edge("underwriter",  END)

    db_path = Path(__file__).parent / "sessions.db"
    thread_id = str(uuid.uuid4())

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        print("\n" + "=" * 54)
        print("  Virtual Insurance Agency — AI Quote System (LangGraph)")
        print("=" * 54)
        print(f"Session: {thread_id}")
        print("Answer the agent's questions to receive a quote.\n")

        graph.invoke(State(), config={"configurable": {"thread_id": thread_id}})


if __name__ == "__main__":
    main()
