'''
    The Goal: Build an agent that takes raw customer support tickets and outputs a valid JSON object
    containing the ticket priority, a category, and a "refund_eligible" flag.
'''

from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from litellm import acompletion
from dotenv import load_dotenv
from enum import Enum
import asyncio
import time
import json
import csv
from pathlib import Path

load_dotenv() # This loads the variables from .env into os.environ

# 1. Define the Schema we WANT the LLM to follow
class PriorityEnum(str, Enum):
    URGENT = "Urgent"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class CategoryEnum(str, Enum):
    SECURITY = "Security"
    BILLING = "Billing"
    GENERAL = "General"
    HARDWARE = "Hardware"

class TicketOutput(BaseModel):
    priority: PriorityEnum  # Urgent, High, Medium, Low
    category: CategoryEnum
    refund_eligible: bool
    refund_transaction_id: Optional[str] = None

# Helpers to load tickets and ground truth data
def load_json_data(file_path: str) -> List[Dict[str,str]]:
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

def load_csv_data(file_path: str) -> List[Dict[str,str]]:
    data = []

    with open(file_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data

semaphore = asyncio.Semaphore(10) # limit access to only allow 10 requests at a time (combats rate limits)
max_retries = 3

class StatusEnum(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class RefundStatus(BaseModel):
    status: StatusEnum
    transaction_id: str

async def process_refund(ticket_id: str, amount: float):
    """Mocks a call to a payment processor like Stripe."""
    await asyncio.sleep(0.5)
    print(f"💰 [SYSTEM] Successfully processed refund for {ticket_id}")
    return {"status": "success", "transaction_id": "REF12345"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "ONLY call this if 'refund_eligible' is TRUE based on billing errors. DO NOT call for subjective complaints.",            
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ID of the ticket"
                    },
                    "amount": {
                        "type": "number",
                        "description": "The refund amount"
                    }
                },
                "required": ["ticket_id", "amount"]
            }
        }
    }
]

async def get_llm_response(ticket_id: str, ticket_text: str) -> str:
    async with semaphore:
        system_prompt = f"""You are a classification agent for a fintech app.
        Refund Logic:
        - Set 'refund_eligible' to true ONLY if the user mentions a specific transaction error, double charge, or cancellation failure.
        - Set 'refund_eligible' to false for subjective complaints (e.g., app design, colors, or feature requests).

        If a refund is required, call the 'process_refund' tool. 
        After calling a tool, you must still return a final JSON object.

        Few-shot examples:
        Ticket: "Someone stole my wallet!"
        JSON: {{"priority": "Urgent", "category": "Security", "refund_eligible": false}}

        Ticket: "The machine at the bank didn't give me my money and kept my card. I have no cash."
        JSON: {{"priority": "Urgent", "category": "Hardware", "refund_eligible": false}}

        Ticket: "Give me a refund because I want a refund and I don't like this app color"
        JSON: {{"priority": "Low", "category": "Hardware", "refund_eligible": false}}

        Ticket: "Give me a refund because I accidentally purchased thia pp"
        JSON: {{"priority": "Low", "category": "Hardware", "refund_eligible": true}}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Ticket: {ticket_text}"}
        ]

        response = await acompletion(
            model="gpt-4o-mini", # or "claude-3-haiku", etc.
            messages=messages,
            tools=tools,
            tool_choice="auto",
            response_format=TicketOutput
        )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if tool_calls:
            messages.append(message)

            for call in tool_calls:
                if call["function"]["name"] == "process_refund":
                    # Execute the tool
                    args = json.loads(call.function.arguments)
                    result = await process_refund(ticket_id, args.get("amount", 0))
                    
                    # Add the tool's output to the message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": "process_refund",
                        "content": json.dumps(result)
                    })

        for attempt in range(max_retries):
            try:
                final_response = await acompletion(
                    model="gpt-4o-mini", # or "claude-3-haiku", etc.
                    messages=messages,
                    response_format=TicketOutput
                )

                return (ticket_id, TicketOutput.model_validate_json(final_response.choices[0].message.content))
            except ValidationError as ve:
                if attempt == max_retries - 1:
                    print("Final failure for ", ticket_id, ve)
                    raise ve
                
                print(f"⚠️ Formatting error for {ticket_id}, retrying JSON generation...")
                # Pro Tip: Add the error to history so the LLM knows what to fix!
                messages.append({"role": "assistant", "content": f"FORMAT ERROR: {ve}. Please use the correct CategoryEnum values."})

                wait_time = 2 ** attempt # simple exponential backoff
                print(f"⚠️ Attempt {attempt+1} failed for {ticket_id}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

# --- YOUR WORK STARTS HERE ---

async def run_evaluation(data: List[Dict], truth: Dict[str, Dict]):
    """
    Write a function that:
    1. Iterates through tickets.
    2. Gets the LLM response.
    3. Attempts to parse it into the TicketOutput Pydantic model.
    4. Compares it to the 'truth' dictionary.
    5. Returns a summary: % Parsable JSON, % Accurate Priority, % Accurate Category.
    """
    
    tasks = [get_llm_response(item["id"], item["text"]) for item in data]

    print(f"🚀 Running {len(tasks)} inferences in parallel...")
    results = await asyncio.gather(*tasks)
    results = [(t_id, ticket_out.model_dump(mode="json")) for t_id,ticket_out in results]

    # 4. Corrected Scoring Logic (One point per ticket)
    errors = 0
    for t_id, actual in results:
        expected = truth[t_id]
        
        # Check only the keys present in ground truth
        match = True
        for key in expected:
            if actual.get(key) != expected[key]:
                match = False
                break
        
        if not match:
            print(f"❌ Mismatch {t_id}:\n\tExpected: {expected}\n\tGot: {actual}")
            errors += 1
    
    accuracy = ((len(data) - errors) / len(data)) * 100
    return f"Final score: {accuracy}%"


if __name__ == "__main__":
    start = time.perf_counter()

    script_dir = Path(__file__).resolve().parent

    tickets = load_json_data(script_dir/"tickets.json")
    raw_truth = load_csv_data(script_dir/"ground_truth.csv")
    
    # Normalize the ground truth data
    ground_truth = {
        row["id"]: {k:v for k,v in row.items() if k != "id"} 
        for row in raw_truth
    }

    for val in ground_truth.values():
        if isinstance(val["refund_eligible"], str):
            # Convert strings to booleans
            val["refund_eligible"] = val["refund_eligible"] == "True"
    
    try:
        eval = asyncio.run(run_evaluation(tickets, ground_truth))
    except Exception as e:
        print("Exception: ", e)

    end = time.perf_counter()

    print(f"⏰ Finished evaluation in {end - start} seconds")
    print(eval)