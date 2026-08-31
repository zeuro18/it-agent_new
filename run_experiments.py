import argparse
import json
import math
import os
import subprocess
import sys
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval", "results")
CONFIGS = [
    "baseline", "tools_only", "dense", "bm25", "hybrid",
    "hybrid_db_verify", "hybrid_no_policy", "hybrid_policy"
]


def paired_sign_test(succ_a: dict, succ_b: dict):
    """Two-sided exact sign test on discordant tasks. succ_a and succ_b map
    task_id to total successes across all runs of their config. Ties are
    dropped; returns None when every task has the same total."""
    n = w = 0
    for tid, count in succ_a.items():
        diff = count - succ_b.get(tid, 0)
        if diff == 0:
            continue
        n += 1
        if diff < 0:
            w += 1
    if n == 0:
        return None
    w = min(w, n - w)
    p = 2 * sum(math.comb(n, k) for k in range(w + 1)) / 2 ** n
    return min(1.0, round(p, 4))


def aggregate_repeats(start_marker: float):
    """Aggregate result files written after start_marker: mean and standard
    deviation of the success rate per config, then a paired sign test of
    each RAG config against tools_only on per-task successes."""
    runs_by_config = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json") or fname == "retrieval_eval.json":
            continue
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.getmtime(path) < start_marker:
            continue
        with open(path, "r") as f:
            data = json.load(f)
        conf = data.get("config")
        if conf in CONFIGS:
            runs_by_config.setdefault(conf, []).append(data)

    print("\nREPEATED-RUN AGGREGATION (task success rate, mean +/- std)")
    succ_counts = {}
    for conf in CONFIGS:
        runs = runs_by_config.get(conf, [])
        if not runs:
            continue
        rate_list = []
        counts = {}
        for data in runs:
            rs = data["results"]
            rate_list.append(100 * sum(1 for r in rs if r["success"]) / len(rs))
            for r in rs:
                counts[r["task_id"]] = counts.get(r["task_id"], 0) + int(r["success"])
        succ_counts[conf] = counts
        mean = sum(rate_list) / len(rate_list)
        if len(rate_list) > 1:
            std = (sum((x - mean) ** 2 for x in rate_list) / (len(rate_list) - 1)) ** 0.5
        else:
            std = 0.0
        print(f"{conf:<12} {mean:5.1f} +/- {std:4.1f}  ({len(runs)} runs)")

    base = succ_counts.get("tools_only")
    if base:
        print("\nPAIRED SIGN TEST vs tools_only (per-task successes across runs)")
        for conf in CONFIGS:
            if conf in ("tools_only", "baseline") or conf not in succ_counts:
                continue
            p = paired_sign_test(succ_counts[conf], base)
            if p is not None:
                verdict = "significant at p<0.05" if p < 0.05 else "not significant"
                print(f"{conf:<12} p = {p:<8} {verdict}")


def run_experiments(fast: bool = True, delay_s: float = 15.0, repeat: int = 1,
                    selected_configs: list = None, simulate_flaky_writes: bool = False):
    configs_to_run = selected_configs or CONFIGS
    print("Starting IT Agent Evaluation Suite...")
    print(f"Mode: {'fast smoke-test' if fast else 'full task bank'} | "
          f"Configs: {', '.join(configs_to_run)} | "
          f"Repeats per config: {repeat} | Inter-run delay: {delay_s}s"
          + (f" | FlakySim: ON" if simulate_flaky_writes else ""))
    print("=" * 50)

    start_marker = time.time()
    total_runs = len(configs_to_run) * repeat

    run_index = 0
    for rep in range(1, repeat + 1):
        for conf in configs_to_run:
            run_index += 1
            label = f"[run {rep}/{repeat}] " if repeat > 1 else ""
            print(f"\n{label}Running config: {conf}")
            start_time = time.time()

            cmd = [sys.executable, "eval/harness.py", "--config", conf]
            if fast:
                cmd.append("--fast")
            if simulate_flaky_writes:
                cmd.append("--simulate-flaky-writes")
            subprocess.run(cmd, check=True)

            elapsed = time.time() - start_time
            print(f"Finished {conf} in {elapsed:.1f}s")

            # Space out runs so Groq's free-tier rate limit from one run
            # doesn't bleed into the next.
            if run_index < total_runs and delay_s > 0:
                print(f"Waiting {delay_s}s before the next run to respect rate limits...")
                time.sleep(delay_s)

    print("\n" + "=" * 50)
    print("All experiments completed.")

    if repeat > 1:
        aggregate_repeats(start_marker)
    else:
        print("Generating comparison metrics...")
        subprocess.run([sys.executable, "eval/metrics.py", "--results", "eval/results/"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full IT Agent experiment suite")
    parser.add_argument("--full", action="store_true",
                        help="Run the full task bank instead of the --fast smoke-test subset")
    parser.add_argument("--config", choices=CONFIGS, help="Run only a specific config")
    parser.add_argument("--simulate-flaky-writes", action="store_true",
                        help="Randomly drop ~10%% of DB commits to test verifier.py catches them")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="Seconds to wait between runs to avoid Groq rate limits (default: 15)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Runs per config; with N>1 prints mean/std and paired sign tests (default: 1)")
    args = parser.parse_args()

    selected = [args.config] if args.config else None
    run_experiments(
        fast=not args.full,
        delay_s=args.delay,
        repeat=max(1, args.repeat),
        selected_configs=selected,
        simulate_flaky_writes=args.simulate_flaky_writes,
    )
