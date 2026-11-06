"""
Prefix-caching test for an SGLang server (RadixAttention).

Start the server with cache reporting on, so responses include cached prefix tokens:

    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --port 30000 --api-key local-test --mem-fraction-static 0.6 --enable-cache-report

Add --disable-radix-cache to the server command to run the same test with prefix
caching off, and save it under another name:

    python prefix_test.py --name prefix_test_cache_off
"""

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from openai import OpenAI

parser = argparse.ArgumentParser(description="Measure prefix caching on an SGLang server.")
parser.add_argument("--name", type=str, default="prefix_test", help="Results file name, saved as results/<name>.json.")
args = parser.parse_args()

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="local-test",
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Repeated content creates a substantial shared prefix.
shared_prefix = (
    "You are an AI inference engineering tutor. "
    "Explain technical concepts accurately. "
    "Discuss GPU memory, batching, and KV caching. "
) * 60

questions = [
    "What is continuous batching?",
    "What is PagedAttention?",
    "What is RadixAttention?",
    "What is KV-cache fragmentation?",
]


def percentile(values, pct):
    # Nearest-rank percentile, so p95 is always a value that was actually observed
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def gpu_memory_mib():
    # Whole-GPU usage from nvidia-smi, including other processes such as the display.
    # SGLang reserves its memory pool at startup, so this stays flat across requests.
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        used, total = (int(value) for value in output.strip().split(","))
        return used, total
    except Exception:
        return None, None


results = []
gpu_peak_mib = 0

for question in questions:
    prompt = shared_prefix + "\nQuestion: " + question

    start = time.perf_counter()
    first_token_time = None
    usage = None

    # Streaming is needed to see when the first token arrives (TTFT)
    stream = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=32,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if first_token_time is None and chunk.choices and chunk.choices[0].text:
            first_token_time = time.perf_counter()
        if chunk.usage is not None:
            usage = chunk.usage

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end

    used_mib, total_mib = gpu_memory_mib()
    if used_mib is not None:
        gpu_peak_mib = max(gpu_peak_mib, used_mib)

    # Only reported when the server runs with --enable-cache-report
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None)

    result = {
        "question": question,
        "ttft_s": round(first_token_time - start, 4),
        "latency_s": round(end - start, 3),
        "prompt_tokens": usage.prompt_tokens,
        "cached_prefix_tokens": cached_tokens,
        "output_tokens": usage.completion_tokens,
        "gpu_memory_used_mib": used_mib,
    }
    results.append(result)
    print(result)

ttfts = [r["ttft_s"] for r in results]
total_output_tokens = sum(r["output_tokens"] for r in results)
total_latency = sum(r["latency_s"] for r in results)
cached = [r["cached_prefix_tokens"] for r in results if r["cached_prefix_tokens"] is not None]

summary = {
    "mean_ttft_ms": round(statistics.mean(ttfts) * 1000, 1),
    "p95_ttft_ms": round(percentile(ttfts, 95) * 1000, 1),
    # Requests run one after another, so this is total output tokens over total request time
    "output_tokens_per_second": round(total_output_tokens / total_latency, 2),
    "cached_prefix_tokens": sum(cached) if cached else None,
    "prompt_tokens": sum(r["prompt_tokens"] for r in results),
    "gpu_memory_used_mib": gpu_peak_mib or None,
    "gpu_memory_total_mib": total_mib,
}

cached_text = (
    f"{summary['cached_prefix_tokens']:,} of {summary['prompt_tokens']:,} prompt tokens"
    if summary["cached_prefix_tokens"] is not None
    else "not reported (start the server with --enable-cache-report)"
)
gpu_text = (
    f"{summary['gpu_memory_used_mib']:,} / {summary['gpu_memory_total_mib']:,} MiB"
    if summary["gpu_memory_used_mib"] is not None
    else "not available (nvidia-smi not found)"
)

print("\n| Metric | Value |")
print("|---|---|")
print(f"| Mean TTFT | {summary['mean_ttft_ms']} ms |")
print(f"| P95 TTFT | {summary['p95_ttft_ms']} ms |")
print(f"| Output tokens/s | {summary['output_tokens_per_second']} |")
print(f"| Cached prefix tokens | {cached_text} |")
print(f"| GPU memory usage | {gpu_text} |")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / f"{args.name}.json").write_text(
    json.dumps({"model": MODEL, "summary": summary, "requests": results}, indent=2),
    encoding="utf-8",
)
print("\nResults saved in:", RESULTS_DIR)
