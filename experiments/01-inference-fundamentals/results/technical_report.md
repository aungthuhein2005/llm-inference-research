
    ## 1. Hardware
    | Property | Value |
    |----------|-------|
    | GPU | Tesla T4 |
    | VRAM (GiB) | 14.56 |
    | CUDA Runtime | 13.0 |
    | Nvidia Driver Version | N/A |
    | PyTorch Version | 2.11.0+cu130 |
    | Transformers Version | 5.20.2 |

    ## 2. Model
    | Property | Value |
    |----------|-------|
    | Model Name | gpt2 |
    | Model Size (Parameters) | 124,439,808 |
    | Weights Precision | torch.float16 |

    ## 3. Workload
    | Property | Value |
    |----------|-------|
    | Input Tokens | 10 |
    | Output Tokens | 10 |
    | Batch Size | 1 |
    | Warmup Iterations | 3 |
    | Measurement Iterations | 5 |

## 4. Prefill

Median prefill latency: **9.46 ms**

## 5. Decode

Median decode latency: **7.79 ms/token**

Median decode throughput: **122.48 tokens/s**

## 6. Memory

| Metric | MiB |
|---|---:|
| Peak allocated VRAM | 498.51 |
| Peak reserved VRAM | 512.00 |
