# 1. SGLang

This experiment measures **prefix caching** in SGLang. SGLang stores the KV cache of earlier prompts in a radix tree (RadixAttention). When a new request starts with the same tokens, it reuses that cache instead of running prefill on them again.

`prefix_test.py` sends 4 requests that share a long system prompt (about 1,385 tokens) and differ only in the final question. It streams each response to measure time to first token (TTFT).

## Setup

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GiB VRAM |
| OS | Windows 11, server in WSL2 (Ubuntu 24.04), client on Windows |
| SGLang | 0.5.23 |
| PyTorch | 2.14.1 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (bfloat16) |
| KV cache pool | 170,905 tokens (~2 GB, from `--mem-fraction-static 0.6`) |

### Launch command

```bash
export CUDA_HOME=~/sglang-env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH

python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --port 30000 --api-key local-test --mem-fraction-static 0.6 \
  --enable-cache-report --attention-backend triton --sampling-backend pytorch \
  > ~/sglang.log 2>&1
```

For the cache-off run, the same command with `--disable-radix-cache` added.

| Argument | Why |
|---|---|
| `CUDA_HOME` | SGLang compiles some kernels on first use. WSL has no system CUDA toolkit, so it points at the `nvcc` that pip installed in the environment. |
| `--enable-cache-report` | Makes responses include `cached_tokens`, so the test can count cached prefix tokens. |
| `--attention-backend triton`, `--sampling-backend pytorch` | Avoid FlashInfer, whose kernels failed to build here. This is slower than SGLang's default kernels, so absolute numbers are not directly comparable with other engines. |
| `> ~/sglang.log 2>&1` | Logging to the terminal froze the server when the Windows terminal paused output. |

### Method

For each configuration, the test was run twice as warmup and the third run was recorded. In the cache-on run, the cache was cleared with `POST /flush_cache` before the recorded run, so its first request is a true cache miss.

# 2. Compare with and without cache

## Summary

| Metric | Cache on | Cache off |
|---|---:|---:|
| Mean TTFT | 103.2 ms | 112.3 ms |
| P95 TTFT | 326.6 ms | 185.0 ms |
| Output tokens/s | 91.89 | 101.11 |
| Cached prefix tokens | 4,155 of 5,555 prompt tokens | 0 of 5,555 prompt tokens |
| GPU memory usage | 3,831 / 6,141 MiB | 3,831 / 6,141 MiB |

## Per request

| Request | Cached tokens (on) | TTFT on | TTFT off | Latency on | Latency off |
|---|---:|---:|---:|---:|---:|
| 1. What is continuous batching? | 0 of 1,388 | 326.6 ms | 185.0 ms | 0.596 s | 0.398 s |
| 2. What is PagedAttention? | 1,385 of 1,389 | 22.7 ms | 84.5 ms | 0.289 s | 0.283 s |
| 3. What is RadixAttention? | 1,385 of 1,389 | 43.2 ms | 99.3 ms | 0.273 s | 0.305 s |
| 4. What is KV-cache fragmentation? | 1,385 of 1,389 | 20.1 ms | 80.3 ms | 0.235 s | 0.280 s |

## Findings

- **Prefix caching works as designed.** Once the shared prompt was cached, requests 2–4 reused 1,385 of 1,389 prompt tokens (99.7%) and only ran prefill on the few tokens of the new question.
- **Cached requests reach the first token about 3× faster.** For requests 2–4, mean TTFT was **28.7 ms with the cache on against 88.0 ms with it off**, saving about 59 ms of prefill per request.
- **The summary averages hide this.** With only 4 requests, the first (uncached) request dominates mean and P95 TTFT. That request was slower in the cache-on run (327 ms against 185 ms). Both runs had 0 cached tokens for it, so the gap is run-to-run variation in the first request after a pause (for example GPU clocks ramping up on a laptop), not a cost of caching. P95 here is simply the first request.
- **End-to-end latency barely changes.** Each request generates 32 tokens, and decoding them takes about 0.2 s, far longer than the ~60 ms prefill that caching saves. Prefix caching matters most for long prompts with short answers, or whenever time to first token matters most, such as chat and agents.
- **Caching needs no extra GPU memory.** Usage is 3,831 MiB either way, because SGLang reserves its KV cache pool at startup and cached prefixes live inside that pool.
- **Output tokens/s is lower with the cache on** only because the slow first request is included in the total time. It is not a decode-speed difference.

## Limitations

- One recorded run per configuration, with 4 requests each, so single requests (especially the first) vary noticeably between runs.
- Requests run one at a time; there is no concurrency.
- A synthetic prompt (the same sentences repeated 60 times).
- Triton attention and PyTorch sampling instead of SGLang's default FlashInfer kernels.

## Reproduce

```bash
python prefix_test.py --name prefix_test_cache_on    # server started without --disable-radix-cache
python prefix_test.py --name prefix_test_cache_off   # server started with --disable-radix-cache
```

Results: `results/prefix_test_cache_on.json` and `results/prefix_test_cache_off.json`.
