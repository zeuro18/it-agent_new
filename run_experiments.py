import os
import subprocess
import time

def run_experiments():
    configs = ["baseline", "tools_only", "dense", "bm25", "hybrid"]
    
    print("Starting IT Agent Evaluation Suite...")
    print("=" * 50)
    
    # We will use the 'fast' flag just to ensure it finishes in a reasonable time during this iteration.
    # The user can run it without --fast for the full 40 tasks.
    for conf in configs:
        print(f"\nRunning config: {conf}")
        start_time = time.time()
        
        # Run harness
        subprocess.run(["python", "eval/harness.py", "--config", conf, "--fast"], check=True)
        
        elapsed = time.time() - start_time
        print(f"Finished {conf} in {elapsed:.1f}s")
        
    print("\n" + "=" * 50)
    print("All experiments completed.")
    print("Generating comparison metrics...")
    subprocess.run(["python", "eval/metrics.py", "--results", "eval/results/"])

if __name__ == "__main__":
    run_experiments()
