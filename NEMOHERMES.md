# NEMOHERMES.md
## Hardened Agentic Stack · Cogni-Brain on DGX Spark
**Author:** Rajendra Rawat (`airawatraj`) · **Date:** May 2026

---

## Overview

The vLLM configuration in this repo gets Nemotron-3-Super-120B running at 23+ TPS.
That is the inference layer. This document covers the **agentic layer** — the
configuration that makes autonomous, long-running agent work possible, stable,
and secure on a single DGX Spark.

NemoHermes, OpenShell, and the underlying sandbox (K3s, Landlock, seccomp) are
NVIDIA products and technologies. This document shares configuration choices,
stability findings, and operational experience — not original engineering.

The benchmark screenshots in the README show 90 minutes of autonomous operation,
60 tool-call iterations, 130K context, and progress updates delivered to Telegram.
This document explains the configuration that made it work.

---

## Why NemoHermes and Not NemoClaw or Bare OpenClaw

Three agent configurations were tested before settling on NemoHermes inside
the OpenShell sandbox.

### Bare OpenClaw (not recommended for sensitive workloads)
Bare-bone OpenClaw without the OpenShell sandbox is not suitable for workloads
involving sensitive data or autonomous code execution. Without sandboxing it runs
with broad filesystem and network access, giving an autonomous agent a direct path
to the host OS. Fine for personal experimentation or throwaway POCs — not for any
workload where data security matters.

### NemoClaw (OpenClaw inside OpenShell sandbox)
NemoClaw runs OpenClaw inside the OpenShell/K3s sandbox — the same kernel-level
isolation described below. The security boundary is real. However, NemoClaw proved
unstable for long-horizon autonomous work during testing.

The most severe failure mode was a context loop where the agent printed `HEARTBEAT_OK`
repeatedly until hitting the maximum token limit, dropped the memory write thread
lock, and crashed the agent process entirely. This happened consistently on overnight,
unattended runs.

### NemoHermes inside OpenShell sandbox (chosen)
NemoHermes operates inside the same OpenShell/K3s sandbox as NemoClaw — identical
kernel-level security boundaries. The choice of NemoHermes over NemoClaw is purely
about **stability**, not security. Both are sandboxed. NemoHermes did not exhibit
the context loop failures observed with NemoClaw.

Key stability advantages observed during testing:
- No context loop failures across dozens of overnight runs
- Automatic retry on stream stalls — important for 130K context sessions where
  mid-stream proxy timeouts occur under sustained load
- Stable memory write thread under sustained autonomous operation
- Native Telegram notification integration for progress updates during unattended runs

---

## OpenShell Sandbox — Kernel-Level Isolation

OpenShell is an NVIDIA product that provides a sandboxed execution environment
for agentic workloads. Both NemoClaw and NemoHermes operate inside this sandbox —
the isolation is not specific to NemoHermes.

The sandbox enforces isolation via three mechanisms:

### Landlock (Filesystem Isolation)
Landlock is a Linux kernel security module that restricts filesystem access at the
kernel level. The agent can only read and write within its designated sandbox
directories. Attempts to access paths outside the sandbox are denied at the kernel
level before the syscall completes.

### Seccomp Profiles (Syscall Filtering)
Seccomp profiles restrict which system calls the agent process can make. Network
calls, process spawning, and privileged operations outside the defined profile are
blocked at the kernel level.

### K3s (Container Orchestration)
The agent workload runs inside K3s, a lightweight Kubernetes distribution, which
provides container namespace isolation — network, PID, and mount namespaces
separate from the host.

Together these three layers provide genuine kernel-enforced isolation around
autonomous code execution — not software-level warnings or policy restrictions.

---

## Critical Configuration — Gateway Timeout

The most common failure mode for long-running agentic tasks is a mid-stream
proxy timeout between NemoHermes and the vLLM endpoint. At 131K context,
generation for complex reasoning prompts can take 20–30 seconds per token batch.
Default gateway timeouts cause the agent to receive an incomplete response and
either retry incorrectly or crash.

**Set the OpenShell gateway timeout to 600 seconds before any serious agentic work:**

```bash
openshell inference set -g nemoclaw --provider compatible-endpoint --model Cogni-Brain --timeout 600
```

Without this, sustained agentic sessions will fail mid-task.

---

## Architecture Summary

```
┌─────────────────────────────────────────────┐
│  NemoHermes Agent Runtime (NVIDIA)          │
│  - Long-horizon task planning               │
│  - Tool call orchestration                  │
│  - Automatic stream retry                   │
│  - Telegram progress notifications          │
├─────────────────────────────────────────────┤
│  OpenShell Sandbox (NVIDIA)                 │
│  - Landlock: kernel filesystem isolation    │
│  - Seccomp: syscall filtering               │
│  - K3s: container namespace isolation       │
│  (shared by both NemoClaw and NemoHermes)   │
├─────────────────────────────────────────────┤
│  vLLM · Cogni-Brain (Nemotron-3-Super-120B) │
│  - 23+ TPS · 131K context · NVFP4 + Marlin  │
│  - See README.md and METHODOLOGY.md         │
└─────────────────────────────────────────────┘
```

---

## Agent Configuration Comparison

| Configuration | Sandbox | Stability for long runs | Suitable for production |
|---|---|---|---|
| Bare OpenClaw | ❌ None | Moderate | No |
| NemoClaw (OpenClaw + OpenShell) | ✅ Yes | ⚠️ Context loop failures observed | No |
| NemoHermes + OpenShell (this setup) | ✅ Yes | ✅ Stable | Yes |

---

## Observed Capabilities vs. Basic Local Agent Setups

| Capability | Typical local agent setup | This configuration |
|---|---|---|
| Long-horizon autonomous work | Unstable | Stable (tested to 90+ min, 60 iterations) |
| Kernel-level code execution isolation | None | Landlock + seccomp via OpenShell |
| Filesystem access control | None | Kernel-enforced |
| Stream timeout resilience | Manual | Automatic retry |
| Overnight unattended operation | Risky | Tested and stable |
| Data sovereignty | Partial | Complete (air-gapped capable) |
| Progress monitoring | None | Telegram notifications |

---

## Known Limitations

**Network friction inside the sandbox:** Heavy file write operations inside the K3s
cluster can trigger `Unexpected Error: tailscale: dial timeout`. The 600-second
gateway timeout resolves most cases. Breaking large tasks into smaller subtasks
also reduces timeout frequency.

**Concurrency ceiling for complex reasoning:** While the vLLM layer scales to 4
concurrent sessions at 55+ total TPS, running multiple simultaneous deep reasoning
prompts through NemoHermes can trigger `torch.AcceleratorError: CUDA error: an
illegal instruction was encountered` in `mamba_mixer2.py`. For sustained autonomous
agent work, limit to a single NemoHermes session.

**OpenShell commercial dependency:** OpenShell is an NVIDIA product and is not
open source. The sandboxing described here depends on OpenShell's runtime.

---

## Feedback

If you have experience with alternative agentic configurations on DGX Spark,
or have found more stable setups — please open an issue or PR.

**Author:** Rajendra Rawat · May 2026