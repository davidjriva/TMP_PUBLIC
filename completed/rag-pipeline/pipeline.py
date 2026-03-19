from dotenv import load_dotenv
import json
from pathlib import Path
import os
import csv
import math
from pydantic import BaseModel, Field

import litellm
from unstructured.partition.pdf import partition_pdf

load_dotenv() # This loads the variables from .env into os.environ

def load_json_data(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

def extract_and_chunk_pdf(pdf_path: Path) -> list[dict]:
    print("Starting PDF element extraction. This may take a while on the first run as models are downloaded...")
    # This identifies titles, tables, and body text
    # The default 'auto' strategy can be slow. Using 'fast' is a quicker alternative.
    elements = partition_pdf(filename=pdf_path, strategy="fast")
    
    print("...PDF extraction complete.")
    
    pdf_chunks = []
    for i, element in enumerate(elements):
        # Create a unique ID for each chunk
        chunk_id = f"{pdf_path.stem}-page-{element.metadata.page_number or 'unknown'}-chunk-{i}"
        
        chunk = {
            "content": element.text,
            "metadata": {
                "source": str(pdf_path.name),
                "id": chunk_id,
                "category": element.category,
                "page_number": element.metadata.page_number,
                # Add other relevant metadata from element.metadata if needed
                # e.g., "coordinates": element.metadata.coordinates,
                # "filename": element.metadata.filename,
            }
        }
        pdf_chunks.append(chunk)
        print(f"PDF Chunk {i+1}: Type={chunk['metadata']['category']} | Page={chunk['metadata']['page_number']} | Text='{chunk['content'][:70]}...'")
    return pdf_chunks

def chunk_json_data(json_data: list[dict]) -> list[dict]:
    print("Starting JSON data chunking...")
    json_chunks = []
    for i, item in enumerate(json_data):
        title = item.get("title", "No Title")
        category = item.get("category", "Uncategorized")
        raw_content = item.get("content", "")
        combined_content = f"Title: {title}\nCategory: {category}\nContent: {raw_content}"
        
        chunk = {
            "content": combined_content,
            "metadata": {
                "source": "data.json",
                "id": item.get("id", f"json-chunk-{i}"),
                "category": item.get("category"),
                "title": item.get("title"),
                "last_updated": item.get("metadata", {}).get("last_updated"),
                "priority": item.get("metadata", {}).get("priority"),
                "tags": item.get("metadata", {}).get("tags", []),
            }
        }
        json_chunks.append(chunk)
        print(f"JSON Chunk {i+1}: ID={chunk['metadata']['id']} | Title='{chunk['metadata']['title'][:70]}...'")
    print("...JSON chunking complete.")
    return json_chunks

def load_embedding_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    
    print(f"Loading embedding cache from {cache_path}...")
    cache = {}
    try:
        with open(cache_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row or len(row) != 2:
                    print(f"Warning: Skipping malformed row {i+1} in cache file.")
                    continue
                chunk_id, embedding_str = row
                try:
                    cache[chunk_id] = json.loads(embedding_str)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping row {i+1} for chunk_id '{chunk_id}' due to invalid embedding format.")
    except Exception as e:
        print(f"Error loading cache file: {e}. Starting with an empty cache.")
        return {}
        
    print(f"Loaded {len(cache)} embeddings from cache.")
    return cache

def save_embedding_cache(cache_path: Path, cache: dict[str, list[float]]):
    print(f"Saving {len(cache)} embeddings to cache at {cache_path}...")
    with open(cache_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for chunk_id, embedding in cache.items():
            writer.writerow([chunk_id, json.dumps(embedding)])
    print("...Cache saved.")

def generate_and_cache_embeddings(
    chunks: list[dict], 
    cache: dict[str, list[float]], 
    cache_path: Path, 
    model: str
) -> dict[str, list[float]]:
    """
    Generates embeddings for chunks not present in the cache, updates the cache, and saves it.
    Returns the updated cache.
    """
    chunks_to_embed = [chunk for chunk in chunks if chunk['metadata']['id'] not in cache]

    if chunks_to_embed:
        print(f"\nFound {len(chunks_to_embed)} new chunks to embed...")
        contents_to_embed = [chunk['content'] for chunk in chunks_to_embed]

        print(f"Calling embedding model '{model}'...")
        try:
            response = litellm.embedding(model=model, input=contents_to_embed)
            new_embeddings = response.data
            
            for chunk, embedding_data in zip(chunks_to_embed, new_embeddings):
                cache[chunk['metadata']['id']] = embedding_data['embedding']
            
            save_embedding_cache(cache_path, cache)
        except Exception as e:
            print(f"\n!!! An error occurred during embedding generation: {e}")
            print("Please ensure your API key (e.g., OPENAI_API_KEY) is set in the .env file and is valid.")
            print("Cache will not be updated for this run.")
    else:
        print("\nAll chunk embeddings are already cached.")
    
    return cache

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calculates the cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    mag1 = math.sqrt(sum(x * x for x in v1))
    mag2 = math.sqrt(sum(x * x for x in v2))
    return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0

def retrieve_chunks(query: str, chunks: list[dict], cache: dict[str, list[float]], model: str, top_k: int = 3) -> list[dict]:
    """Embeds the user query and retrieves the top_k most similar chunks using cosine similarity."""
    response = litellm.embedding(model=model, input=[query])
    query_embedding = response.data[0]['embedding']
    
    scored_chunks = []
    for chunk in chunks:
        chunk_id = chunk['metadata']['id']
        chunk_embedding = cache.get(chunk_id)
        if chunk_embedding:
            score = cosine_similarity(query_embedding, chunk_embedding)
            scored_chunks.append((score, chunk))
    
    # Sort by descending score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def generate_response(query: str, context_chunks: list[dict], llm_model: str) -> str:
    """Synthesizes a response using the LLM and the retrieved context chunks."""
    context_text = "\n\n---\n\n".join([f"Source: {c['metadata']['source']}\n{c['content']}" for c in context_chunks])


    response = litellm.completion(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful and strategic AI assistant. Use the following retrieved context to answer the user's question. If the answer is not explicitly contained in the context, say 'I don't have enough information to answer that.'"},
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}    
        ]
    )
    return response.choices[0].message.content

class EvaluationScore(BaseModel):
    score: int = Field(ge=1, le=5, description="The integer score from 1 to 5 based on accuracy and completeness compared to the ground truth.")

def run_evaluation(chunks: list[dict], cache: dict[str, list[float]], embedding_model: str, llm_model: str):
    """Evaluates the RAG system using an LLM-as-a-judge approach."""
    eval_dataset = [
        {
            "question": "What happens if a 503 error is encountered in the Nexus API?",
            "ground_truth": "The client should implement an exponential backoff strategy starting at 1 second."
        },
        {
            "question": "What is the refund policy for Pro Tiers?",
            "ground_truth": "Refunds are processed within 5-7 business days. Full refund if requested within 14 days and API usage is under 500 tokens."
        },
        {
            "question": "Can I use SMS for 2FA?",
            "ground_truth": "No, SMS 2FA is deprecated as of March 2024 due to SIM-swapping risks."
        }
    ]

    print("\n--- Starting Evaluation ---")
    total_score = 0
    
    for i, item in enumerate(eval_dataset):
        print(f"\nEvaluating Q{i+1}: {item['question']}")
        
        retrieved = retrieve_chunks(item['question'], chunks, cache, embedding_model)
        actual_response = generate_response(item['question'], retrieved, llm_model)
        print(f"Generated Response: {actual_response}")
        
        # LLM-as-a-judge prompt
        eval_prompt = f"""You are an impartial judge evaluating a RAG system's response.
        Question: {item['question']}
        Ground Truth: {item['ground_truth']}
        System Response: {actual_response}
        
        Score the system response from 1 to 5 based on accuracy and completeness compared to the ground truth."""
        
        try:
            score_response = litellm.completion(
                model=llm_model, 
                messages=[{"role": "user", "content": eval_prompt}],
                response_format=EvaluationScore
            )

            score_json = score_response.choices[0].message.content
            score = EvaluationScore.model_validate_json(score_json).score
            
            print(f"Judge Score: {score}/5")
            total_score += score
        except Exception as e:
            print(f"Failed to score: {e}")

    avg_score = total_score / len(eval_dataset)
    print(f"\n--- Evaluation Complete ---")
    print(f"Average Score: {avg_score:.2f} / 5.00")

def chat_loop(chunks: list[dict], cache: dict[str, list[float]], embedding_model: str, llm_model: str):
    print("\n--- RAG Chat Interface ---")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            retrieved_chunks = retrieve_chunks(user_input, chunks, cache, embedding_model)
            print("\n[Thinking - Retrieved Contexts...]")
            for c in retrieved_chunks:
                print(f" - {c['metadata']['id']} (from {c['metadata']['source']})")
                
            response = generate_response(user_input, retrieved_chunks, llm_model)
            print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent

    raw_json_data = load_json_data(script_dir/"data.json")
    json_chunks = chunk_json_data(raw_json_data)

    pdf_chunks = extract_and_chunk_pdf(script_dir/"document.pdf")

    # You now have a unified list of chunks from both sources
    all_chunks = json_chunks + pdf_chunks
    print(f"\nTotal combined chunks from all sources: {len(all_chunks)}")

    # --- Start of embedding and caching logic ---
    CACHE_FILE = script_dir / "cache.csv"
    EMBEDDING_MODEL = "text-embedding-ada-002" # Or another model supported by litellm
    LLM_MODEL = "gpt-4o-mini" # Or any chat model supported by litellm

    # 1. Load existing embeddings from cache
    embedding_cache = load_embedding_cache(CACHE_FILE)

    # 2. Generate and cache embeddings for any new chunks
    embedding_cache = generate_and_cache_embeddings(
        chunks=all_chunks,
        cache=embedding_cache,
        cache_path=CACHE_FILE,
        model=EMBEDDING_MODEL
    )

    print(f"\nTotal embeddings in cache: {len(embedding_cache)}")
    
    # --- CLI Loop ---
    print("\nSelect an action:")
    print("1. Chat with your documents")
    print("2. Run Evaluation Framework")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        chat_loop(all_chunks, embedding_cache, EMBEDDING_MODEL, LLM_MODEL)
    elif choice == '2':
        run_evaluation(all_chunks, embedding_cache, EMBEDDING_MODEL, LLM_MODEL)
    else:
        print("Invalid choice. Exiting.")
    