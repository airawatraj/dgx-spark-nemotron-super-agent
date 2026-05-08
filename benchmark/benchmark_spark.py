#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///
"""
DGX Spark / Nemotron-3-Super Benchmark
Tests TPS, TTFT, and maximum usable context window.
Usage: uv run benchmark_spark.py [--host localhost] [--port 8000] [--model Cogni-Brain]
"""

import argparse
import json
import time
import sys
import threading
import statistics
from datetime import datetime
import requests

# ── Config ────────────────────────────────────────────────────────────────────

COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}

def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"

def header(title):
    line = "─" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")

def result_line(label, value, unit="", color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)} {unit}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_prompt(n_words):
    """Generate a prompt of approximately n_words tokens."""
    base = ("The quick brown fox jumps over the lazy dog. " * 50).split()
    words = (base * ((n_words // len(base)) + 1))[:n_words]
    return " ".join(words) + "\n\nSummarize the above text in one sentence."

def count_tokens_approx(text):
    """Rough token count: ~0.75 words per token."""
    return int(len(text.split()) * 1.33)

def stream_completion(host, port, model, prompt, max_tokens=200, timeout=120, debug=False):
    """
    Stream a single completion. Returns:
    (ttft_ms, tps, total_tokens, full_text, error)
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first = None
    tokens = 0
    full_text = ""
    usage_tokens = None  # actual token count from vLLM usage field

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, None, 0, "", f"HTTP {resp.status_code}: {resp.text[:200]}"

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    if debug:
                        print(f"  RAW: {data[:200]}")
                    try:
                        chunk = json.loads(data)
                        # Capture actual token count from usage field (last chunk)
                        if chunk.get("usage"):
                            usage_tokens = chunk["usage"].get("completion_tokens")
                        delta = chunk["choices"][0]["delta"]
                        if debug and not full_text:
                            print(f"  DELTA KEYS: {list(delta.keys())}")
                            print(f"  DELTA: {delta}")
                        text = delta.get("content", "") or ""
                        think = delta.get("reasoning", "") or ""
                        combined = text + think
                        if combined:
                            if t_first is None:
                                t_first = time.perf_counter()
                            full_text += combined
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        if debug:
                            print(f"  PARSE ERROR: {e} on: {data[:100]}")
                        continue

    except requests.exceptions.Timeout:
        return None, None, 0, "", "Timeout"
    except requests.exceptions.ConnectionError:
        return None, None, 0, "", "Connection refused — is vLLM running on port specified?"
    except Exception as e:
        return None, None, 0, "", str(e)

    t_end = time.perf_counter()

    if t_first is None:
        return None, None, 0, full_text, "No tokens generated"

    ttft_ms = (t_first - t_start) * 1000
    generation_time = t_end - t_first

    # Use actual completion_tokens from usage if available,
    # otherwise estimate from character count (~4 chars/token for English)
    if usage_tokens and usage_tokens > 0:
        tokens = usage_tokens
        token_source = "exact"
    else:
        tokens = max(1, len(full_text) // 4)
        token_source = "estimated"

    tps = tokens / generation_time if generation_time > 0 else 0
    return round(ttft_ms), round(tps, 1), tokens, full_text, None

# ── Test 1: Baseline TPS ──────────────────────────────────────────────────────

def test_baseline_tps(host, port, model, debug=False):
    header("TEST 1 — Baseline TPS (single session, short prompt)")
    prompt = "Explain quantum entanglement in simple terms."
    runs = 3
    results = []

    print(f"  Running {runs} consecutive requests...")
    for i in range(runs):
        ttft, tps, tokens, _, err = stream_completion(host, port, model, prompt, max_tokens=300, debug=debug)
        if err:
            print(c(f"  ✗ Run {i+1} failed: {err}", "red"))
            continue
        results.append((ttft, tps, tokens))
        print(f"  Run {i+1}: TTFT={c(str(ttft)+'ms', 'yellow')}  TPS={c(str(tps), 'green')}  tokens={tokens}")
        time.sleep(1)

    if results:
        avg_tps = round(statistics.mean([r[1] for r in results]), 1)
        avg_ttft = round(statistics.mean([r[0] for r in results]))
        peak_tps = max([r[1] for r in results])
        result_line("Average TPS", avg_tps, "tok/s", "green")
        result_line("Peak TPS", peak_tps, "tok/s", "green")
        result_line("Average TTFT", avg_ttft, "ms", "yellow")
        return avg_tps, peak_tps
    return 0, 0

# ── Test 2: TPS vs Output Length ──────────────────────────────────────────────

def test_tps_vs_length(host, port, model):
    header("TEST 2 — TPS vs Output Length")
    lengths = [50, 150, 300, 600, 1000]
    prompt = "Write a detailed explanation of how transformers work in machine learning."

    print(f"  {'Output tokens'.ljust(18)} {'TPS'.ljust(12)} {'TTFT'}")
    print(f"  {'─'*44}")

    for max_tok in lengths:
        ttft, tps, tokens, _, err = stream_completion(
            host, port, model, prompt, max_tokens=max_tok
        )
        if err:
            print(f"  {str(max_tok).ljust(18)} {c('FAILED: '+err, 'red')}")
        else:
            tps_color = "green" if tps >= 15 else "yellow" if tps >= 10 else "red"
            print(f"  {str(tokens+' tokens' if isinstance(tokens,str) else str(tokens)+' tok').ljust(18)} "
                  f"{c(str(tps)+' tok/s', tps_color).ljust(20)} {ttft}ms")
        time.sleep(1)

# ── Test 3: Concurrent Sessions ───────────────────────────────────────────────

def test_concurrent(host, port, model, max_concurrent=4):
    header("TEST 3 — Concurrent Sessions TPS")
    prompt = "Explain the history of the Roman Empire in detail."
    prompts_list = [
        "Explain the history of the Roman Empire in detail.",
        "Describe how neural networks learn from data.",
        "What are the key principles of thermodynamics?",
        "Explain the causes and effects of the French Revolution.",
    ]

    for n in range(1, max_concurrent + 1):
        results = [None] * n
        errors = []

        def run_request(idx):
            ttft, tps, tokens, _, err = stream_completion(
                host, port, model,
                prompts_list[idx % len(prompts_list)],
                max_tokens=200
            )
            if err:
                errors.append(err)
            else:
                results[idx] = tps

        threads = [threading.Thread(target=run_request, args=(i,)) for i in range(n)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t_start

        valid = [r for r in results if r is not None]
        if valid:
            total_tps = round(sum(valid), 1)
            per_session = round(statistics.mean(valid), 1)
            color = "green" if per_session >= 10 else "yellow" if per_session >= 6 else "red"
            print(f"  {str(n)+' session(s)'.ljust(14)} "
                  f"total={c(str(total_tps)+' tok/s', color).ljust(22)} "
                  f"per-session={c(str(per_session)+' tok/s', color)}")
        else:
            print(f"  {str(n)+' session(s)'.ljust(14)} {c('ALL FAILED: ' + str(errors[0]), 'red')}")

        time.sleep(3)

# ── Test 4: Context Window ────────────────────────────────────────────────────

def test_context_window(host, port, model):
    header("TEST 4 — Context Window Limits")
    print(f"  {c('Testing progressively larger contexts...', 'dim')}")
    print(f"  {'Context tokens'.ljust(20)} {'Result'.ljust(20)} {'TPS'}")
    print(f"  {'─'*50}")

    # Context sizes to test (prompt tokens approximately)
    sizes = [1024, 4096, 8192, 16384, 32768, 65536, 98304, 131072]
    last_working = 0

    for size in sizes:
        prompt = make_prompt(int(size * 0.75))  # words → tokens approximation
        actual_tokens = count_tokens_approx(prompt)

        ttft, tps, gen_tokens, _, err = stream_completion(
            host, port, model, prompt,
            max_tokens=100,
            timeout=180
        )

        if err:
            if "context" in err.lower() or "length" in err.lower() or "exceed" in err.lower():
                status = c("✗ Context exceeded", "red")
            elif "Timeout" in err:
                status = c("✗ Timeout", "red")
            else:
                status = c(f"✗ {err[:25]}", "red")
            print(f"  ~{str(actual_tokens)+' tok':15} {status}")
            break
        else:
            last_working = actual_tokens
            tps_color = "green" if tps >= 12 else "yellow" if tps >= 8 else "red"
            print(f"  ~{str(actual_tokens)+' tok':15} {c('✓ OK', 'green'):20} {c(str(tps)+' tok/s', tps_color)}")
            time.sleep(2)

    if last_working:
        result_line("Max working context", f"~{last_working:,}", "tokens", "green")
    return last_working

# ── Test 5: Memory check ──────────────────────────────────────────────────────

def test_memory_check(host, port):
    header("TEST 5 — vLLM Health & Stats")
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=5)
        result_line("Health endpoint", "OK" if r.status_code == 200 else f"HTTP {r.status_code}",
                    color="green" if r.status_code == 200 else "red")
    except Exception as e:
        result_line("Health endpoint", f"FAILED: {e}", color="red")

    try:
        r = requests.get(f"http://{host}:{port}/metrics", timeout=5)
        if r.status_code == 200:
            lines = r.text.split("\n")
            for line in lines:
                if "gpu_cache_usage_perc" in line and not line.startswith("#"):
                    val = line.split()[-1]
                    pct = round(float(val) * 100, 1)
                    color = "green" if pct < 80 else "yellow" if pct < 95 else "red"
                    result_line("KV cache used", f"{pct}%", color=color)
                if "num_requests_running" in line and not line.startswith("#"):
                    result_line("Requests running", line.split()[-1])
    except Exception:
        result_line("Metrics endpoint", "not available", color="yellow")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(avg_tps, peak_tps, max_context, host, port, model):
    header("SUMMARY")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_line("Timestamp", now)
    result_line("Endpoint", f"http://{host}:{port}")
    result_line("Model", model)
    print()
    result_line("Average TPS (single session)", f"{avg_tps}", "tok/s",
                "green" if avg_tps >= 18 else "yellow" if avg_tps >= 14 else "red")
    result_line("Peak TPS (single session)", f"{peak_tps}", "tok/s",
                "green" if peak_tps >= 20 else "yellow" if peak_tps >= 15 else "red")
    result_line("Max usable context", f"~{max_context:,}" if max_context else "not tested", "tokens")
    print()
    if avg_tps >= 20:
        print(f"  {c('★ Excellent — new config working well', 'green')}")
    elif avg_tps >= 15:
        print(f"  {c('✓ Good — solid improvement over 14 TPS baseline', 'yellow')}")
    elif avg_tps >= 10:
        print(f"  {c('△ Moderate — check for swap or memory pressure', 'yellow')}")
    else:
        print(f"  {c('✗ Below baseline — something is wrong', 'red')}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DGX Spark vLLM setup",
        epilog="Run with: uv run benchmark_spark.py"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default="Cogni-Brain")
    parser.add_argument("--debug", action="store_true",
                        help="Print raw stream chunks for debugging")
    parser.add_argument("--skip-context", action="store_true",
                        help="Skip context window test (faster)")
    parser.add_argument("--skip-concurrent", action="store_true",
                        help="Skip concurrent session test")
    args = parser.parse_args()

    print(f"\n{c('DGX Spark Nemotron-3-Super Benchmark', 'bold')}")
    print(f"{c('Target: ', 'dim')}http://{args.host}:{args.port}  model={args.model}")

    # Quick connectivity check
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        if r.status_code != 200:
            print(c(f"\n✗ vLLM not healthy (HTTP {r.status_code}). Is the container running?", "red"))
            sys.exit(1)
    except Exception as e:
        print(c(f"\n✗ Cannot reach vLLM: {e}", "red"))
        print(c("  Make sure spark-brain container is running and port 8000 is open.", "dim"))
        sys.exit(1)

    print(c("  ✓ vLLM is reachable\n", "green"))

    avg_tps, peak_tps = test_baseline_tps(args.host, args.port, args.model, debug=args.debug)
    test_tps_vs_length(args.host, args.port, args.model)

    if not args.skip_concurrent:
        test_concurrent(args.host, args.port, args.model)

    max_context = 0
    if not args.skip_context:
        max_context = test_context_window(args.host, args.port, args.model)

    test_memory_check(args.host, args.port)
    print_summary(avg_tps, peak_tps, max_context, args.host, args.port, args.model)

if __name__ == "__main__":
    main()