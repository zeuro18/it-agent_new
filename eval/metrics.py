"""
metrics.py
Compute and display evaluation metrics from harness results.
"""

import json
import os
import sys
from collections import Counter, defaultdict


def compute_metrics(run_data: dict) -> dict:
    """Compute aggregate metrics from a harness run."""
    results = run_data["results"]
    total = len(results)

    if total == 0:
        return {"error": "No results to compute"}

    successes = sum(1 for r in results if r["success"])
    failures = total - successes

    # Side effects
    tasks_with_se = sum(1 for r in results if r.get("side_effects"))
    total_se = sum(len(r.get("side_effects", [])) for r in results)

    # Latency
    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else mean_latency

    # Tokens
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    # Groq free tier pricing: ~$0 for free models, estimate $0.05/1M tokens for tracking
    est_cost = total_tokens * 0.00005 / 1000  # Very rough estimate

    # Failure taxonomy
    failure_reasons = Counter()
    for r in results:
        if not r["success"] and r.get("failure_reason"):
            # Simplify the reason into a category
            reason = r["failure_reason"]
            if "not found" in reason.lower():
                failure_reasons["Resource Not Found"] += 1
            elif "already" in reason.lower():
                failure_reasons["Duplicate/Conflict"] += 1
            elif "exception" in reason.lower():
                failure_reasons["Exception/Crash"] += 1
            elif "timeout" in reason.lower():
                failure_reasons["Timeout"] += 1
            elif "agent reported failure" in reason.lower():
                failure_reasons["Agent Self-Reported Failure"] += 1
            else:
                failure_reasons["Other: " + reason[:60]] += 1

    # Per-category breakdown
    by_category = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        if r["success"]:
            by_category[cat]["success"] += 1

    # Per-difficulty breakdown
    by_difficulty = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        diff = r.get("difficulty", 0)
        by_difficulty[diff]["total"] += 1
        if r["success"]:
            by_difficulty[diff]["success"] += 1

    # RAG citation count (for Recall@K estimation)
    tasks_with_citations = sum(1 for r in results if r.get("citations"))
    rag_tasks = [r for r in results if r.get("category") in ("policy_query", "ticket_with_policy")]
    rag_with_citations = sum(1 for r in rag_tasks if r.get("citations"))

    return {
        "total_tasks": total,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": round(successes / total * 100, 1),
        "side_effect_rate": round(tasks_with_se / total * 100, 1),
        "total_side_effects": total_se,
        "mean_latency_s": round(mean_latency, 2),
        "p95_latency_s": round(p95_latency, 2),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 4),
        "failure_taxonomy": dict(failure_reasons),
        "by_category": {k: {"total": v["total"], "success": v["success"],
                            "rate": round(v["success"]/v["total"]*100, 1)}
                        for k, v in by_category.items()},
        "by_difficulty": {k: {"total": v["total"], "success": v["success"],
                              "rate": round(v["success"]/v["total"]*100, 1)}
                          for k, v in sorted(by_difficulty.items())},
        "rag_tasks_total": len(rag_tasks),
        "rag_tasks_with_citations": rag_with_citations,
        "rag_recall_at_5": round(rag_with_citations / len(rag_tasks) * 100, 1) if rag_tasks else 0,
        "total_time_s": run_data.get("total_time_s", 0),
    }


def print_metrics(metrics: dict, config_name: str = ""):
    """Pretty-print metrics to the console."""
    print(f"\nEVALUATION METRICS {config_name}")
    print(f"Task Success Rate:    {metrics['success_rate']:>6}%  ({metrics['success_count']}/{metrics['total_tasks']})")
    print(f"Side Effect Rate:     {metrics['side_effect_rate']:>6}%  ({metrics['total_side_effects']} total)")
    print(f"Mean Latency:         {metrics['mean_latency_s']:>6}s")
    print(f"P95 Latency:          {metrics['p95_latency_s']:>6}s")
    print(f"Total Tokens:         {metrics['total_tokens']:>6}")
    print(f"Estimated Cost:       ${metrics['estimated_cost_usd']:<8}")
    print(f"RAG Recall@5:         {metrics['rag_recall_at_5']:>6}%")

    # Per-category
    print(f"\nPer-Category Breakdown:")
    print(f"Category{' '*17} Success  Total   Rate")
    for cat, data in sorted(metrics["by_category"].items()):
        print(f"{cat:<25} {data['success']:>8} {data['total']:>6} {data['rate']:>5}%")

    # Per-difficulty
    print(f"\nPer-Difficulty Breakdown:")
    print(f"Level{' '*20} Success  Total   Rate")
    for level, data in metrics["by_difficulty"].items():
        label = {1: "L1 Simple", 2: "L2 Medium", 3: "L3 Complex"}.get(level, f"L{level}")
        print(f"{label:<25} {data['success']:>8} {data['total']:>6} {data['rate']:>5}%")

    # Failure taxonomy
    if metrics["failure_taxonomy"]:
        print(f"\nFailure Taxonomy:")
        for reason, count in sorted(metrics["failure_taxonomy"].items(), key=lambda x: -x[1]):
            print(f"[{count}x] {reason}")

    print()


def compare_results(results_dir: str):
    """Load all result files and print a comparison table."""
    files = sorted([f for f in os.listdir(results_dir) if f.endswith(".json")])

    if not files:
        print("No result files found.")
        return

    all_metrics = []
    for fname in files:
        with open(os.path.join(results_dir, fname), "r") as f:
            data = json.load(f)
        metrics = compute_metrics(data)
        metrics["config"] = data.get("config", fname)
        all_metrics.append(metrics)

    # Print comparison table
    print(f"\nEXPERIMENT COMPARISON")
    print(f"{'Config':<15} {'Success%':>9} {'SideEfx%':>9} {'Recall@5':>9} {'Latency':>8} {'Tokens':>8} {'Cost':>8}")

    for m in all_metrics:
        print(f"{m['config']:<15} {m['success_rate']:>8}% {m['side_effect_rate']:>8}% {m['rag_recall_at_5']:>8}% {m['mean_latency_s']:>7}s {m['total_tokens']:>7} ${m['estimated_cost_usd']:>6}")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IT Agent Eval Metrics")
    parser.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--file", help="Single result file to analyze")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            data = json.load(f)
        metrics = compute_metrics(data)
        print_metrics(metrics, data.get("config", ""))
    else:
        compare_results(args.results)
