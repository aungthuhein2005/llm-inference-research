"""
Concurrency benchmark for a vLLM OpenAI-compatible server.

Start the server first (Linux, WSL or Colab; vLLM does not run natively on Windows):

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --api-key local-test

Add --no-enable-prefix-caching to the server command to stop vLLM reusing the
shared prompt between requests, which otherwise makes prefill almost free.
"""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT = "Explain continuous batching in LLM serving."

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def percentile(values, pct):
    # Nearest-rank percentile, so p95 is always a latency that was actually observed
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def request_one(client, model, max_tokens, index):
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
            }
        ],
        temperature=0,
        max_tokens=max_tokens,
        # vLLM-specific: keep generating past end-of-sequence so every request
        # produces exactly max_tokens and runs are comparable
        extra_body={"ignore_eos": True},
    )

    latency = time.perf_counter() - start

    return {
        "index": index,
        "latency_s": latency,
        "output_tokens": response.usage.completion_tokens,
    }


async def benchmark(client, model, max_tokens, concurrency, num_requests):
    # At most `concurrency` requests are in flight at once; the rest wait their turn
    semaphore = asyncio.Semaphore(concurrency)

    async def limited(index):
        async with semaphore:
            return await request_one(client, model, max_tokens, index)

    await request_one(client, model, max_tokens, -1)  # warmup

    start = time.perf_counter()

    results = await asyncio.gather(*[limited(i) for i in range(num_requests)])

    duration = time.perf_counter() - start
    latencies = [result["latency_s"] for result in results]
    total_tokens = sum(result["output_tokens"] for result in results)

    summary = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "duration_s": round(duration, 3),
        "requests_per_second": round(num_requests / duration, 3),
        # Aggregate decode throughput across all in-flight requests: the number
        # continuous batching should push up as concurrency grows
        "output_tokens_per_second": round(total_tokens / duration, 2),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_p50_s": round(percentile(latencies, 50), 3),
        "latency_p95_s": round(percentile(latencies, 95), 3),
        "avg_output_tokens": round(total_tokens / num_requests, 2),
    }

    print(
        f"Concurrency {concurrency:>3} | "
        f"{summary['output_tokens_per_second']:>8.2f} tok/s | "
        f"{summary['requests_per_second']:>6.3f} req/s | "
        f"p50 {summary['latency_p50_s']:.3f} s | "
        f"p95 {summary['latency_p95_s']:.3f} s"
    )

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Benchmark a vLLM server at increasing concurrency.")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", type=str, default="local-test", help="API key the server was started with.")
    parser.add_argument("--model", type=str, default=MODEL, help="Model name served by vLLM.")
    parser.add_argument("--num_requests", type=int, default=32, help="Requests sent at every concurrency level.")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Concurrency levels to test.")
    parser.add_argument("--max_tokens", type=int, default=64, help="Output tokens per request.")
    args = parser.parse_args()

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=120,
    )

    summaries = []
    for concurrency in args.concurrency:
        summaries.append(
            await benchmark(client, args.model, args.max_tokens, concurrency, args.num_requests)
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "benchmark.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    (RESULTS_DIR / "benchmark.json").write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "prompt": PROMPT,
                    "num_requests": args.num_requests,
                    "max_tokens": args.max_tokens,
                },
                "results": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nResults saved in:", RESULTS_DIR)


if __name__ == "__main__":
    asyncio.run(main())
