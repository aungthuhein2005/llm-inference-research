from pathlib import Path
import torch
import transformers
import platform
import subprocess
import time
import argparse
import statistics
import json
import csv

from torch.profiler import profile, ProfilerActivity

# Results and report are written next to this script, regardless of the working directory
EXPERIMENT_DIR = Path(__file__).resolve().parent


def get_gpu_info():
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        ),
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }

    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--id=0",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits"], text=True)
        info["driver_version"] = result.strip()
    except Exception as e:
        info["driver_version"] = f"Error retrieving driver info: {e}"
    return info


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def run_inference(model, input_ids, output_tokens):
    """
    Run greedy generation, timing prefill and each decode step separately.
    Embedding-> Attention -> MLP -> Layer 1 -> Layer 2 -> ... -> Layer N

    Prefill processes the whole prompt in one forward pass and produces the first
    output token. Each decode step then feeds one token plus the KV cache.
    """

    #prefill phase
    synchronize_gpu()
    start = time.perf_counter()

    output = model(input_ids=input_ids, use_cache=True, return_dict=True)

    #cache is used for the next token generation, it contains the key and value tensors for each layer
    cache = output.past_key_values

    #next_token is the greedy choice from the last position's logits, shape (batch, 1)
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    synchronize_gpu()
    prefill_ms = (time.perf_counter() - start) * 1000  # Convert to milliseconds

    #decode phase: prefill already produced the first token, so output_tokens - 1 steps remain
    decode_times = []

    for _ in range(output_tokens - 1):
        synchronize_gpu()
        start = time.perf_counter()

        #run one forward pass on the last token, reusing the cache
        output = model(input_ids=next_token, past_key_values=cache, use_cache=True, return_dict=True)

        #update the cache and next_token for the next iteration
        cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        synchronize_gpu()
        decode_times.append((time.perf_counter() - start) * 1000)  # Convert to milliseconds

    decode_ms_per_token = statistics.mean(decode_times)
    total_ms = prefill_ms + sum(decode_times)

    return {
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "total_ms": total_ms,
        # Decode throughput: tokens generated per second during the decode phase only
        "tokens_per_second": 1000 / decode_ms_per_token,
        # End-to-end throughput: all output tokens over prefill + decode time
        "e2e_tokens_per_second": output_tokens / (total_ms / 1000),
    }


def save_csv(path, runs):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=runs[0].keys())
        writer.writeheader()
        writer.writerows(runs)


def make_report(summary, path):
    h = summary["hardware"]
    m = summary["model"]
    w = summary["workload"]
    t = summary["timing"]
    mem = summary["memory"]

    report = f"""# Experiment 01: Inference Fundamentals

## 1. Hardware
| Property | Value |
|----------|-------|
| GPU | {h['gpu']} |
| VRAM (GiB) | {h['vram_gib']} |
| CUDA Runtime | {h['cuda_runtime']} |
| Nvidia Driver Version | {h['driver_version']} |
| PyTorch Version | {h['pytorch']} |
| Transformers Version | {h['transformers']} |
| Python Version | {h['python']} |

## 2. Model
| Property | Value |
|----------|-------|
| Model Name | {m['id']} |
| Model Size (Parameters) | {m['parameters']:,} |
| Weights Precision | {m['dtype']} |
| Model Type | {m['type']} |

## 3. Workload
| Property | Value |
|----------|-------|
| Input Tokens | {w['input_tokens']} |
| Output Tokens | {w['output_tokens']} |
| Batch Size | {w['batch_size']} |
| Warmup Iterations | {w['warmup_iterations']} |
| Measurement Iterations | {w['measurement_iterations']} |

## 4. Prefill

Median prefill latency: **{t['prefill_median_ms']:.2f} ms**

This measures prompt processing and next-token logits computation.
It approximates time to first token, but excludes tokenization,
network transfer, and serving overhead.

## 5. Decode

Median decode latency: **{t['decode_median_ms_per_token']:.2f} ms/token**

Median decode throughput: **{t['tokens_per_second_median']:.2f} tokens/s**

Median end-to-end throughput (prefill + decode): **{t['e2e_tokens_per_second_median']:.2f} tokens/s**

The measurement includes Python dispatch, synchronization,
argmax selection, and GPU execution.

## 6. Memory

| Metric | MiB |
|---|---:|
| Peak allocated VRAM | {mem['peak_allocated_mib']:.2f} |
| Peak reserved VRAM | {mem['peak_reserved_mib']:.2f} |

## 7. CUDA Kernels

See `results/top_cuda_operations.txt` for the profiler table.

## 8. Bottleneck Hypothesis

**Status: Not yet established.**

Use CUDA kernel timings and additional hardware counters
to determine whether execution is compute-bound,
memory-bandwidth-bound, or mixed.

## 9. Evidence

- Prefill and decode were measured separately.
- Multiple warmed-up measurements were collected.
- CUDA operations were recorded using PyTorch Profiler.
- Peak PyTorch GPU memory was measured.

## 10. Limitations

- Single GPU and single model.
- Batch size fixed at one.
- Synthetic prompt workload.
- Greedy token generation.
- No request scheduling or concurrent users.
- No direct energy measurement.
- No GPU memory-bandwidth or tensor-core utilization counters.
- Python and synchronization overhead affect latency.
- Warm measurements do not represent cold server startup.
- Prefill trace does not include decode operations.

## Conclusion

The experiment establishes a baseline for future
quantization and inference optimization experiments.
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Profile a transformer model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the transformer model to profile.")
    parser.add_argument("--input_tokens", type=int, default=10, help="Number of input tokens for the model.")
    parser.add_argument("--output_tokens", type=int, default=10, help="Number of output tokens to generate.")
    parser.add_argument("--warmup_iterations", type=int, default=5, help="Number of warmup iterations before measurement.")
    parser.add_argument("--measurement_iterations", type=int, default=10, help="Number of measurement iterations to profile.")
    parser.add_argument("--results_dir", type=str, default=str(EXPERIMENT_DIR / "results"), help="Directory for summary.json, runs.csv and profiler output.")
    parser.add_argument("--output_report", type=str, default=str(EXPERIMENT_DIR / "report.md"), help="Path to save the markdown report.")
    args = parser.parse_args()
    MODEL_ID = args.model_name
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")

    if args.input_tokens < 1 or args.output_tokens < 2:
        raise ValueError("Input tokens must be at least 1 and output tokens at least 2 (one prefill token plus one decode step).")
    if args.warmup_iterations < 0 or args.measurement_iterations <= 0:
        raise ValueError("Warmup iterations must be non-negative and measurement iterations must be positive.")

    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting Hardware Information...")
    hardware_info = get_gpu_info()

    print("Loading Model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        ).cuda().eval()
    parameters = sum(p.numel() for p in model.parameters())

    # Fixed synthetic workload: repeated valid token IDs.
    # This ensures the exact requested token length.
    token_id = tokenizer.encode("Hello", add_special_tokens=False)[0]

    input_ids = torch.full(
        (1, args.input_tokens),
        token_id,
        dtype=torch.long,
        device="cuda",
    )

    print(f"Model parameters: {parameters:,}")
    print(f"Input tokens: {args.input_tokens}")

    print("Warming up GPU...")
    for _ in range(args.warmup_iterations):
        run_inference(model, input_ids, args.output_tokens)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    runs = []

    print("Starting benchmark...")
    for i in range(args.measurement_iterations):
        result = run_inference(
            model, input_ids, args.output_tokens
        )
        result = {"run": i + 1, **result}
        runs.append(result)

        print(
            f"Run {i+1}: "
            f"Prefill={result['prefill_ms']:.2f} ms | "
            f"Decode={result['decode_ms_per_token']:.2f} ms/token | "
            f"Speed={result['tokens_per_second']:.2f} tok/s"
        )

    synchronize_gpu()

    memory = {
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2, 2
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2, 2
        ),
    }

    print("Profiling prefill CUDA operations...")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
        synchronize_gpu()

    # Newer PyTorch names this column "self_device_time_total"; older versions use "self_cuda_time_total"
    try:
        profiler_table = prof.key_averages().table(
            sort_by="self_device_time_total",
            row_limit=20,
        )
    except (AttributeError, KeyError, ValueError):
        profiler_table = prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20,
        )

    (result_dir / "top_cuda_operations.txt").write_text(
        profiler_table,
        encoding="utf-8",
    )

    prof.export_chrome_trace(
        str(result_dir / "prefill_trace.json")
    )

    summary = {
        "hardware": hardware_info,
        "model": {
            "id": MODEL_ID,
            "parameters": parameters,
            "dtype": str(next(model.parameters()).dtype),
            "type": model.config.model_type,
        },
        "workload": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "batch_size": 1,
            "warmup_iterations": args.warmup_iterations,
            "measurement_iterations": args.measurement_iterations,
        },
        "timing": {
            "prefill_median_ms": statistics.median(
                r["prefill_ms"] for r in runs
            ),
            "decode_median_ms_per_token": statistics.median(
                r["decode_ms_per_token"] for r in runs
            ),
            "tokens_per_second_median": statistics.median(
                r["tokens_per_second"] for r in runs
            ),
            "e2e_tokens_per_second_median": statistics.median(
                r["e2e_tokens_per_second"] for r in runs
            ),
        },
        "memory": memory,
    }

    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    save_csv(result_dir / "runs.csv", runs)

    make_report(
        summary,
        args.output_report,
    )

    print("\n===== BENCHMARK SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print("\nResults saved in:", result_dir.resolve())
    print("Report saved to:", Path(args.output_report).resolve())


if __name__ == "__main__":
    main()
