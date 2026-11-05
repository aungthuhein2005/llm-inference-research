import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key='local-test'
)

start = time.perf_counter()

response = client.chat.completions.create(
     model="Qwen/Qwen2.5-0.5B-Instruct",
     messages=[
        {
            "role": "user",
            "content": "Explain KV caching.",
        }
     ],
     temperature=0,
     max_tokens=64,
)

end = time.perf_counter() - start

print(f"Response: {response.choices[0].message.content}")
print(f"Latency: {round(end,3)} seconds")
print(f"output tokens: {response.usage.completion_tokens}")
