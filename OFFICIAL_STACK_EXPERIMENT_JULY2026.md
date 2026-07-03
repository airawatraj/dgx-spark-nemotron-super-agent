# OFFICIAL_STACK_EXPERIMENT_JULY2026.md

## Nemotron-3-Super-120B on DGX Spark · July 2026 Official-Stack Re-Test

**Author:** Rajendra Singh Rawat (`airawatraj`)  
**Date:** July 3, 2026  
**Hardware:** NVIDIA DGX Spark — GB10 Grace-Blackwell, 128 GB unified memory  
**Served model name:** `Cogni-Brain`  
**Original methodology:** [`METHODOLOGY.md`](./METHODOLOGY.md)

---

## TL;DR

Does the new public vLLM / DGX Spark guidance change the required setup for
running Nemotron-3-Super-120B-A12B-NVFP4 well on a single DGX Spark?

**No, not materially.**

The official/public path simplifies one painful part of the original setup:
the external `super_v3_reasoning_parser.py` plugin is no longer required.
The reasoning parser is now built into vLLM as:

```bash
--reasoning-parser nemotron_v3
```

Everything else from the original war story still matters.

The best July 2026 result is still the original repo-style configuration:
NGC vLLM image, MARLIN FP4 path, FP8 KV cache, MTP speculative decoding,
conservative memory allocation, explicit batching limits, and
`qwen3_coder` for tool-call parsing.

| Path | Result |
|---|---:|
| Public vLLM / HF quickstart-style path | **16.5 tok/s** |
| Public image + Spark tuning | **22.6 tok/s** |
| NGC image + original repo tuning | **23.6 tok/s avg / 24.3 peak** |
| Tool-eval score on best July config | **90 / 100** |
| Original May 2026 repo score | **93 / 100** |

The conclusion is simple:

> The official stack did not obsolete this repo.  
> It validated the repo.

---

## Background

On June 1, 2026, the vLLM team published:

> [vLLM on the DGX Spark: Architecture, Configuration, and Local Evaluation](https://vllm.ai/blog/2026-06-01-vllm-dgx-spark)

Shortly after, the Hugging Face model card for
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` showed a public vLLM path
using vLLM 0.20.0 and DGX Spark as a supported target.

That raised the obvious question for this repo:

**Did the official/public path finally replace the custom setup documented in
`METHODOLOGY.md`?**

This follow-up answers that question with a fresh July 2026 run.

---

## Timeline

| Date | Event |
|---|---|
| May 9, 2026 | First commit in this repo |
| May 22, 2026 | `METHODOLOGY.md` documents the working DGX Spark stack |
| June 1, 2026 | vLLM DGX Spark blog post published |
| June 2026 | Nemotron-3-Super HF card updated with vLLM 0.20.0 guidance |
| July 3, 2026 | This official-stack re-test run |

This repo documented the working DGX Spark configuration before the later
public documentation became available. The later documentation is useful, but
the working baseline came from the earlier hands-on setup work in this repo.

---

## What Was Tested

Three configurations were tested on the same DGX Spark hardware in sequence.

All runs used:

- Single DGX Spark
- Single-node inference
- No tensor parallelism
- OpenAI-compatible vLLM endpoint on `localhost:8000`
- Served model alias: `Cogni-Brain`
- Max context target: `131072`
- Tool-eval benchmark through the OpenAI-compatible API
- Page cache flush between major runs where applicable:

```bash
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

---

## Config A — Public vLLM / HF Quickstart-Style Path

This is the closest public-image baseline path: vLLM DockerHub image,
default-ish backend selection, no explicit MARLIN tuning, no speculative
decoding.

```bash
docker rm -f spark-brain 2>/dev/null || true

docker run -d --name spark-brain --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.20.0 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.82 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --async-scheduling \
    --enable-chunked-prefill \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Observed path:** public vLLM 0.20.0  
**Backend:** auto-selected path  
**MTP speculative decoding:** not enabled  
**Result:** **16.5 tok/s**

---

## Config B — Public Image + Community Spark Tuning

This keeps the public vLLM image but applies the important tuning discovered
during the original repo work: MARLIN, chunked prefill, explicit batching,
Mamba cache dtype, and MTP speculative decoding.

```bash
docker rm -f spark-brain 2>/dev/null || true

docker run -d --name spark-brain --gpus all \
  --ipc=host \
  --shm-size=16gb \
  -p 8000:8000 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:v0.20.0 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.82 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --async-scheduling \
    --enable-chunked-prefill \
    --moe-backend marlin \
    --mamba-ssm-cache-dtype float32 \
    --max-cudagraph-capture-size 128 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}' \
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

**Observed path:** public vLLM 0.20.0  
**Backend:** MARLIN forced via env + CLI  
**MTP speculative decoding:** enabled, 1 token  
**Result:** **22.6 tok/s**

Tool calling with `hermes` on this public-image path did not reproduce the
original repo's agentic reliability.

---

## Config C — NGC Image + Original Repo Tuning, Updated Parser

This is the original working repo configuration updated for July 2026.

The only meaningful simplification from the original methodology is that the
external parser plugin is no longer needed. The command now uses:

```bash
--reasoning-parser nemotron_v3
```

Everything else remains intentionally close to the original stack.

```bash
docker rm -f spark-brain 2>/dev/null || true

docker run -d --name spark-brain --gpus all \
  --restart=unless-stopped \
  --ipc=host \
  --shm-size=16gb \
  -p 8000:8000 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -v "$HOME/nim-cache:/nim-cache" \
  nvcr.io/nvidia/vllm:26.05-py3 \
  vllm serve /nim-cache/ngc/hub/models--nim--nvidia--nemotron-3-super-120b-a12b/snapshots/rl-030326-nvfp4 \
    --served-model-name Cogni-Brain \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
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
    --reasoning-parser nemotron_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**Observed engine:** `vLLM 0.20.1+7124b12a.dev`  
**Image:** `nvcr.io/nvidia/vllm:26.05-py3`  
**Backend:** MARLIN forced via env + CLI  
**KV cache:** FP8  
**MTP speculative decoding:** enabled, 1 token  
**Tool parser:** `qwen3_coder`  
**Result:** **23.6 tok/s avg / 24.3 peak**  
**SMARTS score:** **90 / 100**

This is the best working NVIDIA-maintained path I found for Nemotron-3-Super
on DGX Spark.

---

## Speed Results

| Config | Image | Avg TPS | Peak TPS | Max Context |
|---|---|---:|---:|---:|
| A — Public quickstart-style path | `vllm/vllm-openai:v0.20.0` | **16.5 tok/s** | 16.5 tok/s | 131,072 |
| B — Public image + Spark tuning | `vllm/vllm-openai:v0.20.0` | **22.6 tok/s** | 22.9 tok/s | 131,072 |
| C — NGC + original repo tuning | `nvcr.io/nvidia/vllm:26.05-py3` | **23.6 tok/s** | 24.3 tok/s | 131,072 |

Config C is stable at roughly 23–24 tok/s across the tested context range.

![Benchmark test 1-3 July 2026](assets/benchmark_test_1-3_july2026.png)

![Benchmark test 4-5 July 2026](assets/benchmark_test_4-5_july2026.png)

---

## Context Window Result

Config C was tested progressively across increasing prompt sizes.

| Approx context | Result | TPS |
|---:|---|---:|
| ~1,030 tokens | OK | 24.4 tok/s |
| ~4,095 tokens | OK | 24.4 tok/s |
| ~8,180 tokens | OK | 24.6 tok/s |
| ~16,352 tokens | OK | 25.1 tok/s |
| ~32,695 tokens | OK | 23.9 tok/s |
| ~65,381 tokens | OK | 24.1 tok/s |
| ~98,067 tokens | OK | 24.1 tok/s |
| ~130,753 tokens | OK | 24.5 tok/s |

**Max working context observed:** approximately **130,753 tokens**

The important part is not just that the server accepts a 131K context setting.
The important part is that decode speed does not collapse at the top of the
window.

---

## Tool-Call Benchmark — SMARTS

Tool-agent behaviour was tested using `tool-eval-bench` v1.7.0.

| Config | Parser | Score | Rating |
|---|---|---:|---|
| B — Public image + Spark tuning | `hermes` | **20 / 100** | Poor |
| C — NGC + original repo tuning | `qwen3_coder` | **90 / 100** | Excellent |
| Original repo, May 2026 | `qwen3_coder` | **93 / 100** | Excellent |

Config C July 2026 category breakdown:

| Category | Score | Earned |
|---|---:|---:|
| Tool Selection | 100% | 6 / 6 |
| Parameter Precision | 67% | 4 / 6 |
| Multi-Step Chains | 100% | 6 / 6 |
| Restraint & Refusal | 100% | 6 / 6 |
| Error Recovery | 83% | 5 / 6 |

Summary:

```text
Score: 90 / 100
Rating: Excellent
Passed: 13
Partial: 1
Failed: 1
Quality: 90 / 100
Responsiveness: 24 / 100
Deployability: 70 / 100
Weakest category: Parameter Precision
```

![SMARTS benchmark July 2026 1](assets/benchmark_smarts_july2026_1.png)

![SMARTS benchmark July 2026 2](assets/benchmark_smarts_july2026_2.png)

![SMARTS benchmark July 2026 3](assets/benchmark_smarts_july2026_3.png)

---

## Key Findings

### 1. The public quickstart-style path is not the performance path

The public vLLM image with the simple/default path reached **16.5 tok/s** in
this run. That is below the original repo baseline.

The model and hardware are capable of more. The difference is configuration.

For DGX Spark, the performance path still requires explicit tuning:

- MARLIN backend
- FP8 KV cache
- MTP speculative decoding
- explicit batch-token cap
- conservative memory allocation
- appropriate Mamba cache dtype

### 2. MARLIN still matters on DGX Spark

The original methodology forced MARLIN because the automatic path was not
reliably selecting the best kernel for this hardware/profile.

That remains true in July 2026.

The tuned public image improved from **16.5 tok/s** to **22.6 tok/s** after
Spark-specific tuning. The NGC image plus original tuning reached **23.6 tok/s
avg / 24.3 tok/s peak**.

### 3. The `qwen3_coder` tool-parser war story is still true

`METHODOLOGY.md` documented `--tool-call-parser qwen3_coder` as the working
compromise for tool calling on this stack.

The July 2026 run confirms the same operational lesson.

The 70-point gap tracks the parser/runtime path: `hermes` on the public image
fails the agentic benchmark, while `qwen3_coder` on the NGC image restores the
behaviour documented in the original methodology.

Observed result:

- `hermes` on public image path: **20 / 100 SMARTS**
- `qwen3_coder` on NGC path: **90 / 100 SMARTS**

For real agent work, throughput is not enough. Tool correctness matters.

### 4. The NGC image is the best working NVIDIA-maintained path found here

The vLLM blog points to a DGX Spark direction. The HF card gives a public
vLLM path. In this local testing, the strongest result came from the
NVIDIA-maintained NGC image:

```text
nvcr.io/nvidia/vllm:26.05-py3
```

That image reported:

```text
vLLM 0.20.1+7124b12a.dev
```

On this DGX Spark, it delivered the best combination of:

- stable 23–24 tok/s decode
- full 131K usable context
- 90/100 tool-eval score
- OpenAI-compatible serving for NemoHermes / local agents

### 5. One real simplification: the parser plugin is gone

The original setup required an external parser plugin:

```text
super_v3_reasoning_parser.py
```

That is no longer needed for this path.

Use the built-in parser:

```bash
--reasoning-parser nemotron_v3
```

This is the one clear quality-of-life improvement from the later stack.

### 6. The FP4 warning is expected, not fatal

The logs may show a warning like:

```text
WARNING: Your GPU does not have native support for FP4 computation.
Weight-only FP4 compression will be used via the Marlin kernel.
```

On DGX Spark, this is expected.

The practical result matters more than the warning: with the MARLIN path,
Config C still reaches stable 23–24 tok/s decode.

---

## What Changed vs Original Repo

| Item | Original May 2026 methodology | July 2026 result |
|---|---|---|
| External reasoning parser plugin | Required | No longer required |
| `--reasoning-parser` | plugin-provided `super_v3` | built-in `nemotron_v3` |
| MARLIN env vars | Required | Still required for best result |
| MTP speculative decoding | Required for peak baseline | Still required for best result |
| FP8 KV cache | Used in tuned config | Still used |
| `--tool-call-parser qwen3_coder` | Required for agent reliability | Still required for best result |
| NGC image | Earlier NVIDIA image / nightly path | `nvcr.io/nvidia/vllm:26.05-py3` |
| Performance | ~24 tok/s | ~23–24 tok/s |
| Tool eval | 93 / 100 | 90 / 100 |

One parser-file workaround disappeared.

The rest of the war story carries forward.

---

## llama-benchy Full Run — Config C

A full Spark Arena-style `llama-benchy` run was also captured for community
reference.

Raw CSV:

```text
assets/llama_benchy_full_july2026_not_submitted.csv
```

> Note: these results are provided as community reference only and have not
> been submitted to Spark Arena. The original repo benchmark results remain
> the authoritative leaderboard submission. Single-session `tg128` in this
> run is slightly below the original result due to run conditions and normal
> run-to-run variance.

Selected results:

| Test | t/s avg | Peak t/s |
|---|---:|---:|
| tg128 c1 baseline | 21.67 ± 0.66 | 25.33 |
| tg128 c2 | 36.98 ± 1.23 | 43.00 |
| tg128 c5 | 35.68 ± 0.58 | 68.67 |
| tg128 c10 | 38.12 ± 0.36 | 66.33 |
| tg128 @ d4096 c1 | 21.98 ± 0.49 | 26.33 |
| tg128 @ d8192 c1 | 22.25 ± 0.37 | 25.67 |
| tg128 @ d16384 c1 | 23.90 ± 0.61 | 26.67 |
| tg128 @ d32768 c1 | 23.00 ± 0.28 | 26.00 |
| tg128 @ d65535 c1 | 22.86 ± 1.38 | 26.33 |
| tg128 @ d100000 c1 | 21.57 ± 0.72 | 24.67 |

The run shows stable single-session decode from small prompts through large
context depths. No meaningful collapse was observed at high context.

---

## Practical Recommendation

If you are running Nemotron-3-Super-120B-A12B-NVFP4 on a single DGX Spark
today, use the Config C shape:

```bash
nvcr.io/nvidia/vllm:26.05-py3
```

with:

```bash
--served-model-name Cogni-Brain
--gpu-memory-utilization 0.75
--max-model-len 131072
--max-num-seqs 4
--max-num-batched-tokens 16384
--kv-cache-dtype fp8
--moe-backend marlin
--quantization fp4
--speculative_config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"triton"}'
--reasoning-parser nemotron_v3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

For production-ish local agent work, do **not** blindly chase the largest
advertised context. The stable, useful target on this stack remains:

```text
131K context
~23–24 tok/s decode
qwen3_coder tool parsing
conservative memory allocation
```

That is the practical sweet spot for Nemotron-3-Super on one DGX Spark.

---

## Relationship to the Next Repo

This project has reached its natural conclusion.

Nemotron-3-Super remains a capable text-only reasoning model with a strong
thinking-trace format. But on the same DGX Spark hardware, my current daily
stack has moved to:

**[`dgx-spark-qwen-omni-super-agent`](https://github.com/airawatraj/dgx-spark-qwen-omni-super-agent)**

That repo documents:

- Qwen3.5-122B on DGX Spark
- 262K context
- ~54 tok/s sustained decode
- 68 tok/s observed peak
- 100/100 tool-eval score
- multimodal voice + image capability
- same single DGX Spark hardware

This Nemotron repo remains useful because it documents the hard path:
how to make a 120B-class reasoning model run locally and reliably on a
single DGX Spark.

The Qwen repo documents the next step:
more speed, more context, better tool reliability, and multimodal capability
on the same machine.

---

## Reproducibility and Comparison Notes

These results are from my DGX Spark, my local runtime conditions, and my
agent/tool workloads. They are not meant to be the final word on
Nemotron-3-Super performance.

I would be happy to be proven wrong.

If you can reproduce better results on a single DGX Spark — especially stable
262K or 1M context, higher sustained decode, or stronger tool-agent behaviour —
please open an issue or PR with:

- exact Docker image and digest
- full launch command
- vLLM version
- model path or checkpoint revision
- memory settings
- parser settings
- benchmark command
- raw benchmark output
- tool-eval results, if applicable

I am especially interested in comparing notes on configurations that beat
23–24 tok/s at 131K context while preserving reliable tool calling.

---

## Conclusion

The original `METHODOLOGY.md` configuration is **not obsolete**.

The July 2026 public/official-stack re-test shows:

- public defaults are easier, but slower
- Spark-specific tuning still matters
- MARLIN still matters
- MTP still matters
- `qwen3_coder` still matters for agent tool use
- the external reasoning parser plugin is no longer needed
- the best result still looks like the original repo config

This repo found the working path first.

The official stack now makes one part cleaner.

The war story still stands.

---

*Benchmarks run July 3, 2026 on NVIDIA DGX Spark, GB10 Grace-Blackwell,
128 GB unified memory. All tests single-node. No tensor parallelism.
Benchmark tooling: `llama-benchy` and `tool-eval-bench` v1.7.0.*
