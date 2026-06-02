"""
benchmark_runner.py
-------------------
The main coordinator script that runs both the LEI pipeline and the LangGraph workflow,
logs comparative CPU/RAM and elapsed time metrics, and writes the results to a CSV file.
"""

import os
import sys
import csv
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

# Setup paths
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))

# Sibling LEI path
LEI_DIR = Path(__file__).resolve().parent.parent.parent / "LEI-LLM-assistedEI"
LEI_DATA_DIR = LEI_DIR / "data"

from metrics import ResourceProfiler
from task_generator import run_lei_task_generator
from code_generator import run_lei_code_generator
from langgraph_workflow import run_langgraph_workflow

def setup_dataset(dataset_name: str) -> Path:
    """Copy dataset files from global LEI data folder to benchmarks dataset folder."""
    src_dir = LEI_DATA_DIR / dataset_name
    dest_dir = BENCHMARKS_DIR / "datasets" / dataset_name
    
    if not src_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder '{src_dir}' not found. "
            "Ensure the LEI-LLM-assistedEI repository is located as a sibling directory."
        )
        
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    files = ["sample_data.csv", "metadata.json", "content.txt"]
    for file in files:
        src_file = src_dir / file
        dest_file = dest_dir / file
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
        else:
            # Handle possible name variance (e.g. content.txt vs context.txt)
            if file == "content.txt":
                alt_file = src_dir / "context.txt"
                if alt_file.exists():
                    shutil.copy2(alt_file, dest_file)
                    continue
            print(f"[WARNING] File '{src_file}' not found during dataset setup.")
            
    print(f"[OK] Setup dataset '{dataset_name}' under: {dest_dir}")
    return dest_dir

def run_lei_benchmark(dataset_dir: Path, output_dir: Path) -> dict:
    """Execute and profile the LEI Step 1 + Step 2 execution flow."""
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK: LEI PIPELINE")
    print("=" * 60)
    
    profiler = ResourceProfiler()
    profiler.start()
    
    # 1. Step 1: Task Generation
    existing_tasks_path = output_dir / "tasks_list.json"
    tasks_data = run_lei_task_generator(dataset_dir, existing_tasks_path)
    
    # Save transient tasks list
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "new_tasks.json", "w", encoding="utf-8") as f:
        json_dump = json.dumps(tasks_data, indent=2) if tasks_data else '{"tasks":[]}'
        f.write(json_dump)
        
    # 2. Step 2: Code Generation
    code_results = run_lei_code_generator(dataset_dir, tasks_data, output_dir / "lei_codes")
    
    stats = profiler.stop()
    
    # Compile performance results
    success_count = sum(1 for c in code_results.values() if c.get("status") == "success")
    failed_count = sum(1 for c in code_results.values() if c.get("status") == "failed")
    
    stats.update({
        "framework": "LEI",
        "tasks_generated": len(tasks_data.get("tasks", [])) if tasks_data else 0,
        "code_success": success_count,
        "code_failed": failed_count
    })
    return stats

def run_langgraph_benchmark(dataset_dir: Path, output_dir: Path) -> dict:
    """Execute and profile the LangGraph node execution flow."""
    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK: LANGGRAPH WORKFLOW")
    print("=" * 60)
    
    profiler = ResourceProfiler()
    profiler.start()
    
    # Invoke stateful graph flow
    graph_output = run_langgraph_workflow(dataset_dir, output_dir / "langgraph_codes")
    
    stats = profiler.stop()
    
    metrics = graph_output.get("metrics", {})
    
    stats.update({
        "framework": "LangGraph",
        "tasks_generated": metrics.get("tasks_count", 0),
        "code_success": metrics.get("success_count", 0),
        "code_failed": metrics.get("failed_count", 0)
    })
    return stats

def save_benchmark_row(results_path: Path, dataset_name: str, stats: dict):
    """Save execution statistics row to comparative CSV file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_path.exists() and results_path.stat().st_size > 0
    
    fieldnames = [
        "timestamp",
        "dataset",
        "framework",
        "tasks_generated",
        "code_generated_success",
        "code_generated_failed",
        "elapsed_time_sec",
        "avg_cpu_percent",
        "avg_memory_mb",
        "peak_memory_mb"
    ]
    
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "framework": stats["framework"],
            "tasks_generated": stats["tasks_generated"],
            "code_generated_success": stats["code_success"],
            "code_generated_failed": stats["code_failed"],
            "elapsed_time_sec": stats["elapsed_time_sec"],
            "avg_cpu_percent": stats["avg_cpu_percent"],
            "avg_memory_mb": stats["avg_memory_mb"],
            "peak_memory_mb": stats["peak_memory_mb"]
        })

def print_summary_comparison(lei_stats: dict, lg_stats: dict):
    """Print comparative metrics summaries in terminal."""
    print("\n" + "=" * 60)
    print("COMPARATIVE BENCHMARKING SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} | {'LEI':<12} | {'LangGraph':<12}")
    print("-" * 60)
    print(f"{'Tasks Generated':<30} | {lei_stats['tasks_generated']:<12} | {lg_stats['tasks_generated']:<12}")
    print(f"{'Code Gen Success':<30} | {lei_stats['code_success']:<12} | {lg_stats['code_success']:<12}")
    print(f"{'Code Gen Failed':<30} | {lei_stats['code_failed']:<12} | {lg_stats['code_failed']:<12}")
    print(f"{'Elapsed Execution Time (s)':<30} | {lei_stats['elapsed_time_sec']:<12} | {lg_stats['elapsed_time_sec']:<12}")
    print(f"{'Average CPU Usage (%)':<30} | {lei_stats['avg_cpu_percent']:<12} | {lg_stats['avg_cpu_percent']:<12}")
    print(f"{'Average Memory (RSS) (MB)':<30} | {lei_stats['avg_memory_mb']:<12} | {lg_stats['avg_memory_mb']:<12}")
    print(f"{'Peak Memory (RSS) (MB)':<30} | {lei_stats['peak_memory_mb']:<12} | {lg_stats['peak_memory_mb']:<12}")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark framework: LangGraph vs LEI.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="agri-data",
        help="Dataset name (e.g. agri-data, air_quality, meteo-data)"
    )
    args = parser.parse_args()
    
    dataset_name = args.dataset
    output_base_dir = BENCHMARKS_DIR / "results" / dataset_name
    results_csv_path = BENCHMARKS_DIR / "results" / "benchmark_results.csv"
    
    # 1. Setup global datasets locally in benchmark folder
    try:
        dataset_dir = setup_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    # 2. Run LEI Benchmark
    lei_stats = run_lei_benchmark(dataset_dir, output_base_dir / "lei")
    save_benchmark_row(results_csv_path, dataset_name, lei_stats)
    
    # 3. Run LangGraph Benchmark
    lg_stats = run_langgraph_benchmark(dataset_dir, output_base_dir / "langgraph")
    save_benchmark_row(results_csv_path, dataset_name, lg_stats)
    
    # 4. Print Summary table
    print_summary_comparison(lei_stats, lg_stats)
    print(f"[OK] comparative benchmark saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
