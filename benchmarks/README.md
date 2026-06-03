# 📊 LangGraph vs. LEI Comparative Benchmarking Suite

This repository branch includes a fully integrated, lightweight comparative benchmarking suite under the [benchmarks/](./) directory.

### ⚙️ How the Comparison Works
*   **LEI Complete Pipeline**: Sequentially drives the 4 baseline execution stages:
    $$\text{task\_generator.py} \rightarrow \text{code\_generator.py} \rightarrow \text{validator.py} \rightarrow \text{edge\_scheduler\_sequential.py}$$
*   **LangGraph Workflow**: Executes the low-level low-latency compiled node sequence:
    $$\text{task\_generation} \rightarrow \text{code\_generation} \rightarrow \text{result\_collection}$$
*   **Metrics Captured**: Profiling elapsed execution time, CPU usage %, average memory (RSS), and peak memory (RSS) using a background thread running at high-frequency (**0.1s interval**).
*   **How to Run**: For complete setup and execution instructions, please refer directly to the [benchmarks/README.md](README.md) file.

---


## ⚙️ How the Comparison Works

The benchmark measures execution resource metrics for both frameworks sequentially under identical parameters:

1.  **LEI Complete Pipeline**: Initiates and profiles the full 4-step execution flow:
    $$\text{task\_generator.py} \rightarrow \text{code\_generator.py} \rightarrow \text{validator.py} \rightarrow \text{edge\_scheduler\_sequential.py}$$
    It runs each script sequentially as a subprocess in its baseline workspace, cleaning up transient folders beforehand, and captures the cumulative hardware metrics.
2.  **LangGraph Workflow**: Runs the stateful compiled node sequence:
    $$\text{task\_generation} \rightarrow \text{code\_generation} \rightarrow \text{result\_collection}$$
    It compiles metrics and writes generated task files locally to `langgraph_codes/`.

---

## 📊 Benchmarking Execution Design

*   **5 Iterations**: The benchmark runs both frameworks **5 times** sequentially.
*   **Clean Baseline**: Before each iteration, previous directories (`generated_tasks/`, `output/`, and `langgraph_codes/`) are fully cleaned to prevent task accumulation or cache interference.
*   **Statistical Analysis (Mean ± Std Dev)**: Each individual run is saved as a row in the output CSV. At the end of the 5 runs, the coordinator automatically computes and prints the **mean ($\mu$) and standard deviation ($\sigma$)** for all metrics, capturing performance and hardware resource variance (important for sudden spikes).
*   **0.1s Background Monitoring**: CPU and memory (RSS) are sampled in a background thread every **0.1 seconds** (rather than 5s point checks) to capture sudden hardware spikes, which is critical for edge deployment on Raspberry Pi.

---

## 🍓 Raspberry Pi 4B & 5 Compatibility
*   **Fully Compatible**: The entire framework is highly optimized for resource-constrained edge environments.
*   **Out-of-the-Box Fallback**: If the standard `langgraph` framework library is missing on your Pi, the workflow compiles and runs seamlessly utilizing a custom pure-Python fallback executor, making it highly robust.

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Set up a Python Virtual Environment
Navigate to the root `langgraph` directory and run:
```bash
# Create venv
python -m venv .venv

# Activate venv (Windows)
.venv\Scripts\activate

# Activate venv (Linux/macOS/Raspberry Pi)
source .venv/bin/activate
```

### Step 2: Install Dependencies
Install the required packages from the benchmarks folder:
```bash
pip install -r benchmarks/requirements.txt
```

### Step 3: Configure Environment Variables
A `.env` file is located at the root of `langgraph/`. The Groq API is active by default. Commented-out support for a local **Ollama** server is also provided.

```env
# Groq API configuration
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
DATA_TYPE=agri-data

# Ollama Configuration (Local LLM on Edge)
# LLM_PROVIDER=ollama
# LLM_BASE_URL=http://http://ip_address/v1
# LLM_API_KEY=ollama
# LLM_MODEL=gemma3:4b
```

To run using Ollama, simply comment out the Groq variables, uncomment the Ollama variables, and save.

### Step 4: Run the Benchmark
Execute the runner specifying the target dataset (e.g., `agri-data`):
```bash
python benchmarks/benchmark_runner.py --dataset agri-data
```

Individual runs will be logged in `benchmarks/results/benchmark_results.csv` and a mean ± standard deviation table will print in the console upon completion.
