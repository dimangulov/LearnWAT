"""
Virtual Insurance Agency — LangGraph version (typed LLM edition)

Same 4-stage pipeline. All LLM I/O goes through Pydantic models:
  - No JSON format instructions in prompts
  - Completion signalled by tool_use, not {"done": true, "data": ...}
  - _converse and _call_once both accept a model_cls and return a typed instance

Graph (linear):
  [receptionist] → [classifier] → [specialist] → [underwriter] → END
"""

import json
import uuid
from pathlib import Path
from typing import TypeVar, Type
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 15
llm = ChatAnthropic(model=MODEL, max_tokens=1024)

# ---------------------------------------------------------------------------
# Pydantic models — all LLM I/O is typed
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


class ClassificationResult(BaseModel):
    product: str   # one of the four product keys
    reason: str    # one sentence


class GeneralLiabilityData(BaseModel):
    clients_on_premises: bool
    handles_property: bool
    prior_claims: int
    state: str
    coverage_limit: str


class ProfessionalLiabilityData(BaseModel):
    services: str
    largest_contract: str
    written_contracts: bool
    prior_claims: int
    coverage_limit: str


class CommercialAutoData(BaseModel):
    vehicle_count: int
    vehicle_types: str
    primary_use: str
    young_drivers: bool
    incidents: int


class WorkersCompData(BaseModel):
    job_classes: str
    states: str
    prior_injuries: int
    uses_subs: bool
    payroll: str


class InsuranceQuote(BaseModel):
    quote_id: str
    product: str
    coverage_limit: str
    annual_premium: int
    monthly_premium: int
    deductible: int
    exclusions: list[str]
    notes: str
    valid_days: int = 30


# Map product key → specialist model class
UNDERWRITING_MODELS: dict[str, type[BaseModel]] = {
    "general_liability":      GeneralLiabilityData,
    "professional_liability": ProfessionalLiabilityData,
    "commercial_auto":        CommercialAutoData,
    "workers_comp":           WorkersCompData,
}

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# System prompts — no JSON format instructions needed
# ---------------------------------------------------------------------------

RECEPTIONIST_PROMPT = """You are a professional insurance agency receptionist.
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
When you have ALL nine items, call the submit_data tool.
"""

CLASSIFIER_PROMPT = """You are an insurance product classifier.
Given a business profile, choose the single best insurance product.

Products:
  general_liability      — physical premises, customers visit, property risk
  professional_liability — advice / consulting / services, errors & omissions
  commercial_auto        — vehicles used for business operations
  workers_comp           — employees doing physical or hazardous work

Call the submit_result tool with your classification.
"""

SPECIALIST_PROMPTS = {
    "general_liability": """You are a General Liability underwriting specialist.
Collect these details in natural conversation:
  1. Do clients visit your premises?
  2. Do you handle third-party property?
  3. Prior claims in the last 3 years?
  4. Primary business state?
  5. Desired coverage limit — $1M, $2M, or $5M?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "professional_liability": """You are a Professional Liability (E&O) underwriting specialist.
Collect in natural conversation:
  1. Specific professional services provided?
  2. Largest single contract value?
  3. Do you use written contracts with all clients?
  4. Prior E&O claims in last 3 years?
  5. Desired coverage limit — $500K, $1M, or $2M?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "commercial_auto": """You are a Commercial Auto underwriting specialist.
Collect in natural conversation:
  1. Number of business vehicles?
  2. Vehicle types (sedans, trucks, vans, etc.)?
  3. Primary use (delivery, sales, transport)?
  4. Any drivers under 25?
  5. Accidents or violations in the last 3 years?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
    "workers_comp": """You are a Workers Compensation underwriting specialist.
Collect in natural conversation:
  1. Job classifications (office, field, construction, etc.)?
  2. States where employees work?
  3. Workplace injuries in last 3 years?
  4. Do you use subcontractors?
  5. Total annual payroll?

Ask 1-2 questions at a time. When you have all five, call the submit_data tool.
""",
}

UNDERWRITER_PROMPT = """You are a senior insurance underwriter.
Given a client profile and underwriting details, produce a realistic quote.
Call the submit_result tool with the completed quote.
"""

# ---------------------------------------------------------------------------
# One-shot chains — prompt template + structured output, no helper needed
# ---------------------------------------------------------------------------

CLASSIFIER_CHAIN = (
    ChatPromptTemplate.from_messages([("system", CLASSIFIER_PROMPT), ("human", "{profile}")])
    | llm.with_structured_output(ClassificationResult)
)

UNDERWRITER_CHAIN = (
    ChatPromptTemplate.from_messages([("system", UNDERWRITER_PROMPT), ("human", "{details}")])
    | llm.with_structured_output(InsuranceQuote)
)

# ---------------------------------------------------------------------------
# State — single source of truth flowing through all nodes
# ---------------------------------------------------------------------------

class State(BaseModel):
    client_data: ClientProfile = ClientProfile()
    product: str = ""
    underwriting_data: dict = {}
    quote: dict = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _converse(system_prompt: str, label: str, model_cls: Type[T]) -> T:
    """
    Conversational loop shared by receptionist and specialist nodes.
    Runs until Claude calls the bound tool (model_cls).
    Returns a typed Pydantic instance — no raw dicts.
    """
    bound_llm = llm.bind_tools([model_cls])
    print(f"\n{'─'*54}\n  {label}\n{'─'*54}\n")
    messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please begin.")]

    for _ in range(MAX_TURNS):
        response = bound_llm.invoke(messages)

        # Tool call = completion signal; return typed instance immediately
        if response.tool_calls:
            print(f"[{label}]: Thank you, I have all the information I need.\n")
            return model_cls(**response.tool_calls[0]["args"])

        # Conversational reply — append AIMessage, prompt user
        messages.append(response)
        print(f"[{label}]: {response.content}\n")
        user_input = input("You: ").strip() or "Please continue."
        messages.append(HumanMessage(content=user_input))

    raise RuntimeError(f"{label}: max turns ({MAX_TURNS}) reached without completing intake — aborting pipeline.")


# ---------------------------------------------------------------------------
# Nodes — each reads from state, returns only what it changes
# ---------------------------------------------------------------------------

def receptionist_node(state: State) -> dict:
    profile = _converse(RECEPTIONIST_PROMPT, "Receptionist", ClientProfile)
    print(f"[System]: Profile collected → {profile.model_dump()}\n")
    return {"client_data": profile}


def classifier_node(state: State) -> dict:
    result = CLASSIFIER_CHAIN.invoke({"profile": state.client_data.model_dump_json()})
    product = result.product
    if product not in SPECIALIST_PROMPTS:
        print(f"[System]: WARNING — unknown product '{product}', defaulting to general_liability.")
        product = "general_liability"
    print(f"[System]: Routing to {product.replace('_', ' ').title()} specialist.")
    print(f"[Reason]: {result.reason}\n")
    return {"product": product}


def specialist_node(state: State) -> dict:
    prompt = SPECIALIST_PROMPTS.get(state.product, SPECIALIST_PROMPTS["general_liability"])
    label = f"{state.product.replace('_', ' ').title()} Specialist"
    model_cls = UNDERWRITING_MODELS.get(state.product, GeneralLiabilityData)
    data = _converse(prompt, label, model_cls)
    print(f"[System]: Underwriting data collected → {data.model_dump_json()}\n")
    # Store as dict — State.underwriting_data is untyped (four possible shapes)
    return {"underwriting_data": data.model_dump()}


def underwriter_node(state: State) -> dict:
    print("[System]: Generating quote...\n")
    quote = UNDERWRITER_CHAIN.invoke({"details": json.dumps({
        "client": state.client_data.model_dump(),
        "product": state.product,
        "underwriting_details": state.underwriting_data,
    })})

    # Merge client info into quote dict for display and persistence
    quote_dict = quote.model_dump()
    quote_dict["client"] = state.client_data.model_dump()

    print("=" * 54)
    print("  INSURANCE QUOTE")
    print("=" * 54)
    print(json.dumps(quote_dict, indent=2))
    print("=" * 54)

    qid = quote.quote_id.replace("/", "-")
    out_path = Path(__file__).parent / f"quote_{qid}.json"
    with open(out_path, "w") as f:
        json.dump(quote_dict, f, indent=2)
    print(f"\nSaved to: {out_path}")

    return {"quote": quote_dict}

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
