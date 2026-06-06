"""
benchmark_runner.py
-------------------
The main coordinator script that runs both the complete 4-step LEI pipeline
and the LangGraph workflow 5 times, logs individual metrics to a CSV file,
and outputs comparative statistics (mean and standard deviation).
"""

import os
import sys
import csv
import json
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

# Setup paths
BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))

# Sibling LEI path
LEI_DIR = Path(__file__).resolve().parent.parent.parent / "LEI-LLM-assistedEI"
LEI_DATA_DIR = LEI_DIR / "data"

# Explicitly load .env at top-level before other modules import config
try:
    from dotenv import load_dotenv
    # Load from local repo root .env first, fallback to LEI root .env
    local_env = BENCHMARKS_DIR.parent / ".env"
    if local_env.exists():
        load_dotenv(local_env)
    else:
        load_dotenv(LEI_DIR / ".env")
except ImportError:
    pass

from metrics import ResourceProfiler
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

def clean_lei_directories(dataset_name: str, keep_tasks_list: bool = True):
    """Clean generated_tasks and output folders in LEI directory to ensure clean runs."""
    generated_dir = LEI_DIR / "generated_tasks" / dataset_name
    output_dir = LEI_DIR / "output" / dataset_name
    
    # Clean output JSONs
    if output_dir.exists():
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
            
    # Clean generated tasks directory (keep tasks_list.json if specified)
    if generated_dir.exists():
        for file_path in generated_dir.glob("*"):
            if file_path.is_file():
                if keep_tasks_list and file_path.name == "tasks_list.json":
                    continue
                try:
                    file_path.unlink()
                except Exception:
                    pass
def _run_lei_script_in_process(script_path: Path, env_override: dict) -> int:
    import sys
    import os
    import importlib
    import io
    
    # Save old environment
    old_env = os.environ.copy()
    
    # Save old sys.path
    old_sys_path = list(sys.path)
    
    # Save old CWD
    old_cwd = os.getcwd()
    
    # Save old stdout/stderr to restore later
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect stdout/stderr to silence screen prints from LEI
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        # Change CWD to LEI_DIR
        os.chdir(str(LEI_DIR))

        # Update environment variables
        for k, v in env_override.items():
            os.environ[k] = str(v)
            
        # Add LEI dir to sys.path
        lei_dir_str = str(LEI_DIR)
        if lei_dir_str not in sys.path:
            sys.path.insert(0, lei_dir_str)
            
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
            
        # Reload key config and utility modules
        for mod_name in ['config', 'shared_utils']:
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                except Exception as e:
                    pass
                    
        # Load the script dynamically
        module_name = script_path.stem
        sys.modules.pop(module_name, None)
        
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"Cannot load script at {script_path}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if hasattr(module, "main"):
            ret = module.main()
        else:
            ret = 0
            
        if ret is None:
            ret = 0
        return ret
    finally:
        # Restore environment, sys.path, CWD, and stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.environ.clear()
        os.environ.update(old_env)
        sys.path = old_sys_path
        os.chdir(old_cwd)


def run_lei_benchmark(dataset_name: str, run_id: str, run_num: int) -> dict:
    """Execute and profile the complete 4-step LEI pipeline sequentially in-process."""
    print(f"\n[LEI Pipeline] Run {run_num}/2 starting...")
    
    # Clean the LEI workspace first to prevent task accumulation (keep tasks_list.json after run 1)
    clean_lei_directories(dataset_name, keep_tasks_list=(run_num > 1))
    
    profiler = ResourceProfiler()
    profiler.start()
    
    import time
    lei_time_up_to_codegen = 0.0
    lei_time_val = 0.0
    
    t_start = time.time()
    try:
        env = {
            "DATA_TYPE": dataset_name,
            "RUN_ID": run_id,
            "RUN_COUNT": str(run_num),
        }
        
        # Step 1: task_generator.py
        ret1 = _run_lei_script_in_process(LEI_DIR / "task_generator.py", env)
        if ret1 != 0:
            raise RuntimeError(f"task_generator.py failed with return code {ret1}")
        
        # Step 2: code_generator.py
        ret2 = _run_lei_script_in_process(LEI_DIR / "code_generator.py", env)
        if ret2 != 0:
            raise RuntimeError(f"code_generator.py failed with return code {ret2}")
            
        lei_time_up_to_codegen = time.time() - t_start
        
        # Step 3: validator.py
        t_val_start = time.time()
        ret3 = _run_lei_script_in_process(LEI_DIR / "validator.py", env)
        lei_time_val = time.time() - t_val_start
        if ret3 != 0:
            raise RuntimeError(f"validator.py failed with return code {ret3}")
        
        # Step 4: edge_scheduler_sequential.py
        ret4 = _run_lei_script_in_process(LEI_DIR / "scheduler" / "edge_scheduler_sequential.py", env)
        if ret4 != 0:
            raise RuntimeError(f"edge_scheduler_sequential.py failed with return code {ret4}")
    finally:
        stats = profiler.stop(label=f"LEI_run{run_num}")
    
    # Extract code status counts from LEI filesystem directories
    generated_dir = LEI_DIR / "generated_tasks" / dataset_name
    
    try:
        with open(generated_dir / "tasks_list.json", "r", encoding="utf-8") as f:
            tasks_list_data = json.load(f)
            tasks_generated = len(tasks_list_data.get("tasks", []))
    except Exception:
        tasks_generated = 0
        
    code_success = len(list(generated_dir.glob("*.py")))
    code_failed = max(0, tasks_generated - code_success)
    
    # Copy LEI generated outputs to benchmark results directory
    lei_results_dir = BENCHMARKS_DIR / "results" / dataset_name / "lei"
    lei_tasks_dest = lei_results_dir / "lei_tasks"
    
    lei_tasks_dest.mkdir(parents=True, exist_ok=True)
    
    # Copy generated python files and task json files
    if generated_dir.exists():
        for file_path in generated_dir.glob("*"):
            if file_path.is_file():
                try:
                    shutil.copy2(file_path, lei_tasks_dest / file_path.name)
                except Exception:
                    pass
    
    stats.update({
        "framework": "LEI",
        "tasks_generated": tasks_generated,
        "code_success": code_success,
        "code_failed": code_failed,
        "lei_time_up_to_codegen": lei_time_up_to_codegen,
        "lei_time_val": lei_time_val
    })
    return stats

def run_langgraph_benchmark(dataset_dir: Path, output_dir: Path, run_num: int) -> dict:
    """Execute and profile the LangGraph node execution flow."""
    print(f"\n[LangGraph Workflow] Run {run_num}/2 starting...")
    
    langgraph_codes_dir = output_dir / "langgraph_codes"
    dataset_name = dataset_dir.name
    
    if run_num == 1:
        # Full clean at the start of the first run
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass
        output_dir.mkdir(parents=True, exist_ok=True)
        langgraph_codes_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Keep tasks_list.json but clean Python files and new_tasks.json inside langgraph_codes
        if langgraph_codes_dir.exists():
            for file_path in langgraph_codes_dir.glob("*"):
                if file_path.is_file():
                    if file_path.name == "tasks_list.json":
                        continue
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
        else:
            langgraph_codes_dir.mkdir(parents=True, exist_ok=True)
            
    profiler = ResourceProfiler()
    profiler.start()
    
    try:
        # Invoke stateful graph flow
        graph_output = run_langgraph_workflow(dataset_dir, langgraph_codes_dir)
        
        # Clean up output JSON results of LangGraph from LEI directory
        lei_output_dir = LEI_DIR / "output" / dataset_name
        if lei_output_dir.exists():
            for file_path in lei_output_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()  # Clean up to avoid leaking to next run
                    except Exception:
                        pass
    finally:
        stats = profiler.stop(label=f"LangGraph_run{run_num}")
    
    metrics = graph_output.get("metrics", {}) if 'graph_output' in locals() else {}
    
    stats.update({
        "framework": "LangGraph",
        "tasks_generated": metrics.get("tasks_count", 0),
        "code_success": metrics.get("success_count", 0),
        "code_failed": metrics.get("failed_count", 0),
        "lei_time_up_to_codegen": 0.0,
        "lei_time_val": 0.0
    })
    return stats


def save_benchmark_row(results_path: Path, dataset_name: str, stats: dict):
    """Save execution statistics row to comparative CSV file."""
    try:
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
            "lei_time_up_to_codegen",
            "lei_time_val",
            "avg_cpu_percent",
            "avg_memory_mb"
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
                "lei_time_up_to_codegen": stats.get("lei_time_up_to_codegen", 0.0),
                "lei_time_val": stats.get("lei_time_val", 0.0),
                "avg_cpu_percent": stats["avg_cpu_percent"],
                "avg_memory_mb": stats["avg_memory_mb"]
            })
    except Exception as e:
        print(f"[WARNING] Could not save benchmark row to {results_path}: {e}")

def calculate_mean_std(values: list) -> tuple:
    """Calculate mean and standard deviation of a numerical list."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5
    return round(mean, 2), round(std, 2)

def print_summary_comparison(lei_results: list, lg_results: list):
    """Print comparative metrics summaries in terminal using mean ± std."""
    # Extract values
    lei_tasks = [r["tasks_generated"] for r in lei_results]
    lg_tasks = [r["tasks_generated"] for r in lg_results]
    
    lei_success = [r["code_success"] for r in lei_results]
    lg_success = [r["code_success"] for r in lg_results]
    
    lei_failed = [r["code_failed"] for r in lei_results]
    lg_failed = [r["code_failed"] for r in lg_results]
    
    lei_time = [r["elapsed_time_sec"] for r in lei_results]
    lg_time = [r["elapsed_time_sec"] for r in lg_results]
    
    lei_up_to_codegen_times = [r.get("lei_time_up_to_codegen", 0.0) for r in lei_results]
    lei_val_times = [r.get("lei_time_val", 0.0) for r in lei_results]
    
    lei_cpu = [r["avg_cpu_percent"] for r in lei_results]
    lg_cpu = [r["avg_cpu_percent"] for r in lg_results]
    
    lei_mem = [r["avg_memory_mb"] for r in lei_results]
    lg_mem = [r["avg_memory_mb"] for r in lg_results]
    
    # Calculate stats
    tasks_lei_m, tasks_lei_s = calculate_mean_std(lei_tasks)
    tasks_lg_m, tasks_lg_s = calculate_mean_std(lg_tasks)
    
    suc_lei_m, suc_lei_s = calculate_mean_std(lei_success)
    suc_lg_m, suc_lg_s = calculate_mean_std(lg_success)
    
    fail_lei_m, fail_lei_s = calculate_mean_std(lei_failed)
    fail_lg_m, fail_lg_s = calculate_mean_std(lg_failed)
    
    time_lei_m, time_lei_s = calculate_mean_std(lei_time)
    time_lg_m, time_lg_s = calculate_mean_std(lg_time)
    
    lei_codegen_m, lei_codegen_s = calculate_mean_std(lei_up_to_codegen_times)
    lei_val_m, lei_val_s = calculate_mean_std(lei_val_times)
    
    cpu_lei_m, cpu_lei_s = calculate_mean_std(lei_cpu)
    cpu_lg_m, cpu_lg_s = calculate_mean_std(lg_cpu)
    
    mem_lei_m, mem_lei_s = calculate_mean_std(lei_mem)
    mem_lg_m, mem_lg_s = calculate_mean_std(lg_mem)
    
    print("\n" + "=" * 70)
    print(f"COMPARATIVE BENCHMARKING SUMMARY (2 RUNS: Mean ± Std Dev)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'LEI':<16} | {'LangGraph':<16}")
    print("-" * 70)
    print(f"{'Tasks Generated':<30} | {tasks_lei_m:<5} ± {tasks_lei_s:<8} | {tasks_lg_m:<5} ± {tasks_lg_s:<8}")
    print(f"{'Code Gen Success':<30} | {suc_lei_m:<5} ± {suc_lei_s:<8} | {suc_lg_m:<5} ± {suc_lg_s:<8}")
    print(f"{'Code Gen Failed':<30} | {fail_lei_m:<5} ± {fail_lei_s:<8} | {fail_lg_m:<5} ± {fail_lg_s:<8}")
    print(f"{'Elapsed Execution Time (s)':<30} | {time_lei_m:<5} ± {time_lei_s:<8} | {time_lg_m:<5} ± {time_lg_s:<8}")
    print(f"{'  - Up to Code Gen (LEI only)':<30} | {lei_codegen_m:<5} ± {lei_codegen_s:<8} | {'N/A':<16}")
    print(f"{'  - Validation (LEI only)':<30} | {lei_val_m:<5} ± {lei_val_s:<8} | {'N/A':<16}")
    print(f"{'Average CPU Usage (%)':<30} | {cpu_lei_m:<5} ± {cpu_lei_s:<8} | {cpu_lg_m:<5} ± {cpu_lg_s:<8}")
    print(f"{'Average Memory (RSS) (MB)':<30} | {mem_lei_m:<5} ± {mem_lei_s:<8} | {mem_lg_m:<5} ± {mem_lg_s:<8}")
    print("=" * 70 + "\n")
 
    summary_path = BENCHMARKS_DIR / "results" / "benchmark_summary.csv"
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["metric", "lei_mean", "lei_std", "langgraph_mean", "langgraph_std"]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            metrics_data = [
                ("tasks_generated", tasks_lei_m, tasks_lei_s, tasks_lg_m, tasks_lg_s),
                ("code_generated_success", suc_lei_m, suc_lei_s, suc_lg_m, suc_lg_s),
                ("code_generated_failed", fail_lei_m, fail_lei_s, fail_lg_m, fail_lg_s),
                ("elapsed_time_sec", time_lei_m, time_lei_s, time_lg_m, time_lg_s),
                ("lei_time_up_to_codegen", lei_codegen_m, lei_codegen_s, 0.0, 0.0),
                ("lei_time_val", lei_val_m, lei_val_s, 0.0, 0.0),
                ("avg_cpu_percent", cpu_lei_m, cpu_lei_s, cpu_lg_m, cpu_lg_s),
                ("avg_memory_mb", mem_lei_m, mem_lei_s, mem_lg_m, mem_lg_s)
            ]
            for row in metrics_data:
                writer.writerow({
                    "metric": row[0],
                    "lei_mean": row[1],
                    "lei_std": row[2],
                    "langgraph_mean": row[3],
                    "langgraph_std": row[4]
                })
        print(f"[OK] Statistical summary saved to: {summary_path}")
    except Exception as e:
        print(f"[WARNING] Could not save benchmark summary to {summary_path}: {e}")

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
    
    # Pre-clean the comparative results CSV to ensure alternating rows start fresh
    if results_csv_path.exists():
        try:
            results_csv_path.unlink()
        except Exception:
            pass
            
    # 1. Setup global datasets locally in benchmark folder
    try:
        dataset_dir = setup_dataset(dataset_name)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
        
    lei_results = []
    lg_results = []
    
    # Unique Run ID for correlating log events of this 2-run block
    run_id = "benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nStarting comparative benchmarking (2 iterations)...")
    for i in range(1, 3):
        print(f"\n--- ITERATION {i}/2 ---")
        
        # 2. Run LEI Pipeline (Task Gen -> Code Gen -> Validator -> Scheduler)
        try:
            lei_stats = run_lei_benchmark(dataset_name, run_id, i)
            save_benchmark_row(results_csv_path, dataset_name, lei_stats)
            lei_results.append(lei_stats)
        except Exception as e:
            print(f"[ERROR] LEI run {i} failed: {e}")
            
        # 3. Run LangGraph Workflow (task nodes -> code nodes -> result end nodes)
        try:
            lg_stats = run_langgraph_benchmark(dataset_dir, output_base_dir / "langgraph", i)
            save_benchmark_row(results_csv_path, dataset_name, lg_stats)
            lg_results.append(lg_stats)
        except Exception as e:
            print(f"[ERROR] LangGraph run {i} failed: {e}")
            
    # 4. Print Summary table
    print_summary_comparison(lei_results, lg_results)
    print(f"[OK] comparative benchmark saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
