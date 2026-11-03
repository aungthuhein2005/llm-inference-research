import fastapi
import asyncio
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
from pydantic import BaseModel
import time

app = fastapi.FastAPI()

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

if torch.cuda.is_available():
    device = "cuda"
else:
    raise RuntimeError("GPU is required for this demo. Please run on a GPU machine.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map=device).eval()

lock = asyncio.Lock()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/generate")
async def generate(request: GenerateRequest):
    async with lock:
        start = time.perf_counter()
        message = [
            {"role": "user", "content": request.prompt}
        ]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()  # Wait for GPU to finish
        input_length = inputs["input_ids"].shape[1]

        generated_text = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        end = time.perf_counter()
        print(f"Generation time: {end - start:.2f} seconds")
        return {"generated_text": generated_text}


