# METHODOLOGY.md
## DGX Spark · Nemotron-3-Super-120B · Single-Node Benchmark
**Author:** Raj Rawat (`airawatraj`) · **Date:** May 8, 2026

---

## Setup

### Hardware
- **Device:** NVIDIA DGX Spark (GB10 Grace-Blackwell Superchip)
- **Unified Memory:** 128 GB (CPU + GPU shared pool)
- **Storage:** EXT4 local NVMe
- **Thermal:** GPU operating under 75°C throughout all runs

### Software Stack
| Component | Version / Reference |
|---|---|
| vLLM image | `vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776` (cu130-nightly) |
| vLLM version | `0.19.2rc1.dev134+gfe9c3d6c5` |
| Model checkpoint | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`rl-030326-nvfp4`) |
| Reasoning parser | `super_v3_reasoning_parser.py` (official NVIDIA HuggingFace release) |
| Agent Runtime | `NemoHermes` (Proved vastly more stable for real workflows than NemoClaw/OpenClaw) |
| Open WebUI | Running alongside during all benchmark runs |
| OS | Ubuntu 24.04 (DGX OS) |

### Final vLLM Launch Command
```bash
docker run -d --name spark-brain --gpus all \
  --restart=unless-stopped \
  --shm-size=16gb \
  -p 8000:8000 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_HUB_OFFLINE=1 \
  -v "$HOME/nim-cache:/nim-cache" \
  -v "$(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py" \
  vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776 \
    --model /nim-cache/ngc/hub/models--nim--nvidia--nemotron-3-super-120b-a12b/snapshots/rl-030326-nvfp4 \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 4 \
    --max-model-len 131072 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

---

## Key Configuration Decisions & War Stories

Getting 120B parameters running natively on a 128GB unified memory system required fighting through several undocumented roadblocks. Here is why the configuration above looks the way it does.

### Why raw vLLM instead of the standard NVIDIA NIM Wrapper?
If you attempt to use the standard NIM container, the process will fatally crash during weight loading. The GB10 chip utilizes a Unified Memory Architecture (UMA). The NIM wrapper has a hardcoded "safety guard" that forcibly clamps `gpu_memory_utilization` from your requested 0.90 down to 0.50 to prevent kernel panics. 
Because 50% of 128GB is ~60.8GB, and the NVFP4 model requires ~75GB just for weights, the NIM wrapper creates a synthetic OOM error and dies immediately. Deploying the raw `vllm-openai` image acts as a "nuclear override" to bypass this wrapper and allow 0.75 utilization.

### The Tool Parser Booby-Trap (`--tool-call-parser qwen3_coder`)
Early runs using the default `hermes` parser or standard OpenAI tool flags crashed the vLLM server entirely. Inspecting the logs revealed a hardcoded stub inside the NVIDIA vLLM fork (`error=Not being used, manual parsing in serving_chat.py`). 
I was caught in a Catch-22: passing no flags resulted in 400 errors from the agent rejecting the payload; passing standard tool flags hit the broken internal parser and triggered a 500 server crash. The golden compromise was passing `--enable-auto-tool-choice` to open the API door, combined with `--tool-call-parser qwen3_coder` to seamlessly route the tool calls to the agent without tripping the buggy internal chat serving code.

### Why `--gpu-memory-utilization 0.75`
The model weights confirmed at 75.03 GiB at load time. At 0.75 utilization:
- 128 GB × 0.75  = 96 GB reserved by vLLM
- Model weights  = 75 GB
- KV cache budget = ~21 GB
- Free headroom  = ~26 GB (Leaves room for OS, OpenShell K3s clusters, and Tailscale networking).
Attempting to push this to 0.82+ resulted in kernel-level lockups when combined with MTP speculative decoding.

### Why `--max-num-batched-tokens 16384`
Setting this to match the context length (131072) caused a CUDA OOM during MoE workspace buffer allocation (`torch.ops.vllm.moe_forward_shared` tried to allocate 43.31 GiB). 16384 is the practical upper bound for the Marlin MoE workspace on GB10.

### Why not 1M Context? & Why Swap is Disabled?
I tested pushing the context window to the heavily marketed 1M-token limit. It actually worked beautifully — the engine burst past **38+ TPS**. But in real-world usage, it resulted in a complete system crash. I **value stability over benchmark rankings**, so I scaled it back to a stable 24 TPS at 65K. NemoClaw/NemoHermes started complaining, though, so I eventually found 131K to be the sweet spot.

The math is unforgiving: for a 120B model, the KV cache footprint for 1M tokens vastly exceeds the ~26 GB of available unified memory headroom on a single Spark. Once the memory filled, the OS aggressively began swapping tensors to the local NVMe SSD. The PCIe bandwidth choke starved the GPU, causing vLLM to hang and eventually crashing the system. 

This is exactly why OS swap must be permanently disabled (`sudo swapoff -a`) in this setup, and why the context is strictly capped at 131K for single-node stability. True 1M context for this model requires a dual-Spark configuration.

---

## Hardware Limitation — No Native FP4 Compute

The GB10 in DGX Spark does **not** have native FP4 tensor core support. The vLLM startup log confirms:
`WARNING: Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel.`

NVFP4 provides **memory compression benefits** (model fits in 128 GB unified memory) but **not native FP4 compute acceleration**. The Marlin kernel dequantizes weights to BF16 for computation. This means B200/H100 benchmarks showing massive NVFP4 compute speedups are not directly comparable to single-Spark results.

---

## Benchmark Results

All results use exact `completion_tokens` from vLLM's streaming usage API to calculate true decode throughput (including generated `<think>` tokens).

### Single Session TPS

| Run | TTFT | TPS | Output tokens |
|---|---|---|---|
| 1 | 211ms | 23.5 | 300 |
| 2 | 218ms | 23.6 | 300 |
| 3 | 212ms | 22.6 | 300 |
| **Average** | **214ms** | **23.2** | |

### Concurrent Sessions

| Sessions | Total TPS | Per-session TPS |
|---|---|---|
| 1 | 22.4 | 22.4 |
| 2 | 35.4 | 17.7 |
| 3 | 41.7 | 13.9 |
| 4 | 55.3 | 13.8 |

4-session total TPS of 55.3 demonstrates strong batching efficiency on GB10.

### Context Window

TPS remains stable at 23+ tokens/s across the full 131K context window. No performance cliff observed.

---

## Known Limitations & Operational Hazards

**Mamba-2 NVFP4 Kernel Sync Errors:** While basic requests scale to 4 concurrent sessions nicely, hitting the engine simultaneously with highly complex, back-to-back reasoning prompts can trigger `torch.AcceleratorError: CUDA error: an illegal instruction was encountered` in `mamba_mixer2.py`. The bleeding-edge Blackwell Tensor Cores occasionally trip over themselves interleaving math for massive parallel thoughts. For deep autonomous agent work, limiting concurrency provides essential stability.

**Agent Sandboxing Network Friction:** Heavy agentic frameworks operating inside privileged sandboxes (like K3s/OpenShell) can choke during massive file write operations. I frequently encountered `Unexpected Error: tailscale: dial timeout` and mid-stream proxy timeouts until I hardcoded the OpenShell gateway timeout to 600 seconds (`openshell inference set --timeout 600`). 

**Context Loops (NemoClaw):** During early testing, NemoClaw occasionally fell into severe repetitive loops (e.g., printing `HEARTBEAT_OK` until it hit the maximum token limit), dropping the memory write thread lock and crashing the agent. Switching the active agent stack to `NemoHermes` proved drastically more stable for overnight, unattended app building.

**Uncalibrated FP8 KV cache scaling.** vLLM logs confirm `Using KV cache scaling factor 1.0` (uncalibrated). This may cause subtle accuracy degradation on very long contexts compared to properly calibrated future releases.

---

## Comparison With Prior Published Results

| Who | TPS | Stack | Context | Concurrent | Production services |
|---|---|---|---|---|---|
| **This work** | **23.2** | NVFP4 + vLLM | 131K | 4 | NemoHermes + Open WebUI |
| Eugr | 16.55 | NVFP4 + vLLM | 256K | unknown | none documented |
| josephbreda | 16–17 | NVFP4 + vLLM | unknown | 1 | none |
| Avarok | 19 | NVFP4 + vLLM | unknown | 1 | none |
| Saiyam Pathak | 19.5 | Q4_K_M GGUF + llama.cpp | 262K | 1 | none |

*If you reproduce these results or find errors in this methodology, please open an issue or pull request. The goal is accurate, reproducible community benchmarks — not records.*