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

if __name__ == "__main__":
    load_cache() # 1. Try to load from disk
    
    # 2. Process Files
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir/"data"
    data = []
    for fp in ["greek.json", "mixed.json"]:
        if (data_dir/fp).exists():
            data += load_json_file(data_dir/fp)

    # 3. Chunk and Embed
    chunks = get_recursive_chunks(data)
    # Always update CHUNK_LOOKUP in case content changed, but skip embeddings if IDs match
    for c in chunks:
        if c['metadata']['id'] not in CHUNK_LOOKUP:
            CHUNK_LOOKUP[c['metadata']['id']] = c

    asyncio.run(run_ingestion(chunks))
    chat_loop()