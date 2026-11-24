import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

question = "Explain Generative AI in simple words."

print("Gemini is responding:\n")

start_time = time.perf_counter()
first_chunk_time = None

response = client.models.generate_content_stream(
    model="gemini-3.5-flash",
    contents=question,
)

last_chunk = None
for chunk in response:
    now = time.perf_counter()
    if first_chunk_time is None and chunk.text:
        first_chunk_time = now
    last_chunk = chunk
    if chunk.text:
        print(chunk.text, end="", flush=True)

end_time = time.perf_counter()
print("\n\nResponse completed!")

# Response time metrics
ttft = (first_chunk_time - start_time) if first_chunk_time else None
total_time = end_time - start_time

print("\n--- Response Time ---")
if ttft is not None:
    print(f"Time to first chunk: {ttft:.2f}s")
print(f"Total stream time:   {total_time:.2f}s")

# Metadata from the final chunk
if last_chunk and last_chunk.usage_metadata:
    usage = last_chunk.usage_metadata
    print("\n--- Metadata ---")
    print("Prompt tokens:", usage.prompt_token_count)
    print("Output tokens:", usage.candidates_token_count)
    print("Total tokens:", usage.total_token_count)

    if usage.candidates_token_count and total_time > 0:
        tps = usage.candidates_token_count / total_time
        print(f"Throughput:          {tps:.1f} tokens/sec")

if last_chunk and last_chunk.candidates:
    candidate = last_chunk.candidates[0]
    print("Finish reason:", candidate.finish_reason)