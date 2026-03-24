"""
Virtual Insurance Agency — Text Agent Prototype

Sequential multi-agent pipeline that produces an insurance quote without human underwriters.

Agents:
  1. Receptionist      (conversational) — collects business profile
  2. Classifier        (internal, 1-shot) — routes to the right product
  3. Product Specialist (conversational) — gathers underwriting details
  4. Underwriter       (internal, 1-shot) — produces structured JSON quote
"""

import json
import re
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-haiku-4-5-20251001"
MAX_TURNS = 15

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RECEPTIONIST_PROMPT = """You are a professional insurance agency receptionist.
Collect the following from the client through natural conversation:
  1. Business name
  2. Owner / contact name
  3. What the business does (2-3 sentences)
  4. Number of employees
  5. Approximate annual revenue

Ask 1-2 questions at a time. Be friendly but concise.

When you have ALL five items, output ONLY this JSON and nothing else:
{"done": true, "data": {"business_name": "...", "owner_name": "...", "business_type": "...", "employees": <int>, "revenue": "..."}}
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
# Agent helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def call_internal(system_prompt: str, user_message: str) -> dict:
    """One-shot internal agent — no user interaction."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    result = _extract_json(response.content[0].text)
    return result or {}


def run_conversational_agent(system_prompt: str, agent_label: str) -> dict:
    """
    Conversational agent loop.
    Agent speaks first, user responds, repeat until agent returns terminal JSON.
    Returns the parsed 'data' dict from the terminal JSON.
    """
    print(f"\n{'─'*54}")
    print(f"  {agent_label}")
    print(f"{'─'*54}\n")

    # Seed: ask agent to open the conversation
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

        # Check for terminal JSON
        parsed = _extract_json(reply)
        if parsed and parsed.get("done"):
            print(f"[{agent_label}]: Thank you, I have all the information I need.\n")
            return parsed.get("data", {})

        # Not done — show reply, get user input
        print(f"[{agent_label}]: {reply}\n")
        user_input = input("You: ").strip()
        if not user_input:
            user_input = "Please continue."
        messages.append({"role": "user", "content": user_input})

    print("WARNING: max turns reached — incomplete data collected.")
    return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*54)
    print("  Virtual Insurance Agency — AI Quote System")
    print("="*54)
    print("Answer the agent's questions to receive a quote.\n")

    # --- Stage 1: Receptionist collects business profile ---
    client_data = run_conversational_agent(RECEPTIONIST_PROMPT, "Receptionist")
    if not client_data:
        print("ERROR: Could not collect client profile.")
        return
    print(f"\n[System]: Profile collected → {json.dumps(client_data)}\n")

    # --- Stage 2: Classifier picks the product (internal) ---
    classification = call_internal(
        CLASSIFIER_PROMPT,
        f"Classify this business for insurance: {json.dumps(client_data)}",
    )
    product = classification.get("product", "general_liability")
    reason = classification.get("reason", "")
    product_label = product.replace("_", " ").title()
    print(f"[System]: Routing to {product_label} specialist.")
    print(f"[Reason]: {reason}\n")

    # --- Stage 3: Product specialist gathers underwriting details ---
    specialist_prompt = SPECIALIST_PROMPTS.get(product, SPECIALIST_PROMPTS["general_liability"])
    underwriting_data = run_conversational_agent(specialist_prompt, f"{product_label} Specialist")
    if not underwriting_data:
        print("ERROR: Could not collect underwriting details.")
        return
    print(f"\n[System]: Underwriting data collected → {json.dumps(underwriting_data)}\n")

    # --- Stage 4: Underwriter produces the final quote (internal) ---
    print("[System]: Generating quote...\n")
    payload = {
        "client": client_data,
        "product": product,
        "underwriting_details": underwriting_data,
    }
    quote = call_internal(UNDERWRITER_PROMPT, json.dumps(payload))

    if not quote:
        print("ERROR: Underwriter could not produce a quote.")
        return

    # Display quote
    print("=" * 54)
    print("  INSURANCE QUOTE")
    print("=" * 54)
    print(json.dumps(quote, indent=2))
    print("=" * 54)

    # Save to file
    qid = quote.get("quote_id", "draft").replace("/", "-")
    filename = f"insurance-prototype/quote_{qid}.json"
    with open(filename, "w") as f:
        json.dump(quote, f, indent=2)
    print(f"\nSaved to: {filename}")


if __name__ == "__main__":
    main()
