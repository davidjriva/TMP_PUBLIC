import pickle
import os
from pathlib import Path
import json
import asyncio
import math
import heapq
import litellm
import tiktoken
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import deque

load_dotenv()

# --- Configuration ---
CACHE_FILE = "mythology_cache.pkl"
VECTOR_STORE = {} 
CHUNK_LOOKUP = {}

# --- Cache Management ---

def save_cache():
    """Saves the current state of vectors and chunks to disk."""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"vectors": VECTOR_STORE, "chunks": CHUNK_LOOKUP}, f)
    print(f"💾 Cache saved to {CACHE_FILE}")

def load_cache():
    """Loads existing vectors and chunks from disk if they exist."""
    global VECTOR_STORE, CHUNK_LOOKUP
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            VECTOR_STORE = data.get("vectors", {})
            CHUNK_LOOKUP = data.get("chunks", {})
        print(f"✅ Loaded {len(VECTOR_STORE)} chunks from cache.")
    else:
        print("📂 No cache found. Starting fresh ingestion.")

# --- Existing Core Logic (Chunking & Similarity) ---

def load_json_file(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_recursive_chunks(documents: list[dict], chunk_size=512, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    def token_len(text): return len(tokenizer.encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_chunks = []
    for doc in documents:
        header = f"Subject: {doc['title']} | Category: {doc['category']} | Content: "
        segments = splitter.split_text(doc['content'])
        for i, segment in enumerate(segments):
            chunk_id = f"{doc['id']}-chunk-{i}"
            final_chunks.append({
                "content": header + segment,
                "metadata": {**doc.get("metadata", {}), "id": chunk_id}
            })
    return final_chunks

def cosine_similarity(v1, v2):
    dot_product = sum(x*y for x,y in zip(v1,v2))
    norm_v1 = math.sqrt(sum(x**2 for x in v1))
    norm_v2 = math.sqrt(sum(y**2 for y in v2))
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0.0

# --- Asynchronous Ingestion with Cache Check ---

async def create_single_embedding(chunk: dict, semaphore: asyncio.Semaphore):
    chunk_id = chunk['metadata']['id']
    
    # 🛑 THE CACHE CHECK
    if chunk_id in VECTOR_STORE:
        return # Skip API call if we already have it

    async with semaphore:
        try:
            response = await litellm.aembedding(model="text-embedding-ada-002", input=chunk['content'])
            VECTOR_STORE[chunk_id] = response.data[0]['embedding']
            CHUNK_LOOKUP[chunk_id] = chunk
        except Exception as e:
            print(f"Failed to embed {chunk_id}: {e}")

async def run_ingestion(chunks: list[dict]):
    sem = asyncio.Semaphore(20)
    # Only process chunks NOT in our current VECTOR_STORE
    new_chunks = [c for c in chunks if c['metadata']['id'] not in VECTOR_STORE]
    
    if not new_chunks:
        print("✨ All chunks are already cached. Skipping embedding phase.")
        return

    print(f"🚀 Embedding {len(new_chunks)} new chunks...")
    tasks = [asyncio.create_task(create_single_embedding(c, sem)) for c in new_chunks]
    await asyncio.gather(*tasks)
    save_cache() # Save once all new embeddings are done

# --- RAG Inference & UI ---

def get_top_k_chunks(query: str, k: int = 5) -> list[dict]:
    query_vec = litellm.embedding(model="text-embedding-ada-002", input=query).data[0]['embedding']
    top_k_heap = []
    for c_id, vec in VECTOR_STORE.items():
        score = cosine_similarity(query_vec, vec)
        heapq.heappush(top_k_heap, (score, c_id))
        if len(top_k_heap) > k: heapq.heappop(top_k_heap)
    
    top_k_heap.sort(key=lambda x: x[0], reverse=True)
    return [CHUNK_LOOKUP[c_id] for _, c_id in top_k_heap]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_top_k_chunks",
            "description": "Gets the top K chunks on Roman, Greek, or Chinese mythology",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

SHORT_TERM_MEMORY = deque()
MAX_SHORT_TERM_MEMORY_SIZE = 5

def chat_loop():
    print("\n🔱 MYTHOLOGY AGENT (CACHED) | Type 'exit' to quit.")

    available_functions = {
        "get_top_k_chunks": get_top_k_chunks    
    }
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ['exit', 'quit']: break
    
        messages = [
            {"role": "system", "content": "You are a Senior Mythologist. Use ONLY provided context. If unsure, say you don't know. You have one tool available called 'get_top_k_chunks' that can be used to retrieve information from the knowledge base on Chinese, Greek, and Roman mythology."},
            {"role": "user", "content": f"Query: {query}"},
            {"role": "user", "content": f"Previous responses: {SHORT_TERM_MEMORY}"}
        ]

        response = litellm.completion(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            messages.append(response.choices[0].message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_response = function_to_call(query)

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    }
                )

                print(f"\n🔧 Agent called tool {function_name}")

            second_response = litellm.completion(
                model="gpt-4o-mini",
                messages=messages,
            )  # get a new response from the model where it can see the function response

            SHORT_TERM_MEMORY.append(second_response.choices[0].message.content)

            print("\nAgent:", second_response.choices[0].message.content)
        else:
            SHORT_TERM_MEMORY.append(response.choices[0].message.content)

            print(f"\nAgent: {response.choices[0].message.content}")

        if len(SHORT_TERM_MEMORY) > MAX_SHORT_TERM_MEMORY_SIZE:
            SHORT_TERM_MEMORY.popleft() # Evict last memory


'''
    EVALUATION CODE
'''

# --- 1. The Golden Dataset ---
# In a real scenario, you'd have 50+ of these.
EVAL_DATASET = [
    {
        "query": "Who is the Roman god of the sea?",
        "expected_tool": "get_top_k_chunks",
        "ground_truth_keywords": ["Neptune", "trident", "water"],
        "expected_culture": "Roman"
    },
    {
        "query": "What are the 12 labors of Heracles?",
        "expected_tool": "get_top_k_chunks",
        "ground_truth_keywords": ["Nemean Lion", "Hydra", "Cerberus"],
        "expected_culture": "Greek"
    },
    {
        "query": "Who is the Monkey King?",
        "expected_tool": "get_top_k_chunks",
        "ground_truth_keywords": ["Sun Wukong", "Journey to the West", "staff"],
        "expected_culture": "Chinese"
    },
    {
        "query": "Hello, how are you today?",
        "expected_tool": None, # Should NOT trigger a search
        "ground_truth_keywords": [],
        "expected_culture": None
    }
]

# --- 2. The Judge Prompt ---
JUDGE_SYSTEM_PROMPT = """
You are an expert Evaluator for RAG systems. You will be given a User Query, 
the Retrieved Context, and the Agent's Final Response.
Score the response from 1-5 on these metrics:

1. FAITHFULNESS: Is the answer derived ONLY from the context? (1 = Hallucinated, 5 = Perfectly Grounded)
2. RELEVANCY: Does it directly answer the user's question?
3. TOOL USE: Did the agent use the mythology tool correctly or unnecessarily?

Output your evaluation in strict JSON format:
{
    "faithfulness_score": int,
    "relevancy_score": int,
    "tool_use_score": int,
    "reasoning": "string"
}
"""

async def run_evaluation_suite(rag_logic_func):
    """
    rag_logic_func: A function that takes a query and returns 
    (final_response, tool_called, retrieved_chunks)
    """
    results = []
    
    print(f"🧪 Starting Evaluation on {len(EVAL_DATASET)} test cases...")

    for test in EVAL_DATASET:
        print(f"Testing: '{test['query']}'")
        
        # 1. Run the system
        response_text, tool_used, chunks = await rag_logic_func(test['query'])
        
        # 2. Call the "Judge" (using a stronger model like GPT-4o)
        context_text = "\n".join([c['content'] for c in chunks])
        
        eval_payload = {
            "query": test['query'],
            "context": context_text,
            "response": response_text
        }

        judge_resp = litellm.completion(
            model="gpt-4o", # Use a 'smarter' model to judge the 'smaller' model
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(eval_payload)}
            ],
            response_format={ "type": "json_object" }
        )
        
        score = json.loads(judge_resp.choices[0].message.content)
        
        # 3. Check Tool Selection logic
        tool_correct = (tool_used == test['expected_tool'])
        score["tool_selection_correct"] = tool_correct
        
        results.append({
            "query": test['query'],
            "scores": score
        })

    # --- 4. Print Summary ---
    print("\n" + "="*30)
    print("📊 EVALUATION SUMMARY")
    print("="*30)
    for r in results:
        s = r['scores']
        status = "✅" if s['tool_selection_correct'] else "❌"
        print(f"{status} Query: {r['query']}")
        print(f"   | Faithfulness: {s['faithfulness_score']}/5")
        print(f"   | Relevancy: {s['relevancy_score']}/5")
        print(f"   | Reason: {s['reasoning']}\n")

# --- Integration Logic ---
async def agent_inference_wrapper(query):
    """A version of your chat_loop without the 'input()' for automation."""
    messages = [
        {"role": "system", "content": "You are a Senior Mythologist..."},
        {"role": "user", "content": query}
    ]
    
    # Import your tools and functions here
    response = litellm.completion(model="gpt-4o-mini", messages=messages, tools=tools)
    
    tool_calls = response.choices[0].message.tool_calls
    tool_name = tool_calls[0].function.name if tool_calls else None
    retrieved_chunks = []
    
    if tool_calls:
        # Call your existing get_top_k_chunks
        retrieved_chunks = get_top_k_chunks(query)
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "name": tool_name,
            "content": json.dumps(retrieved_chunks)
        })
        
        second_resp = litellm.completion(model="gpt-4o-mini", messages=messages)
        return second_resp.choices[0].message.content, tool_name, retrieved_chunks
    
    return response.choices[0].message.content, tool_name, []

if __name__ == "__main__":
    # Ensure cache is loaded before running
    # load_cache() 
    asyncio.run(run_evaluation_suite(agent_inference_wrapper))