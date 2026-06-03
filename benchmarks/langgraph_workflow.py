"""
langgraph_workflow.py
---------------------
Stateful graph workflow utilizing LangGraph.
Features a robust lightweight fallback engine to ensure absolute compatibility 
with Raspberry Pi and other resource-constrained platforms out-of-the-box.
"""

import os
import sys
import json
from pathlib import Path
from typing import TypedDict, Dict, Any, List
from openai import OpenAI
from string import Template

# Add sibling LEI-LLM-assistedEI directory to path (append to avoid name conflicts)
LEI_DIR = Path(__file__).resolve().parent.parent.parent / "LEI-LLM-assistedEI"
if str(LEI_DIR) not in sys.path:
    sys.path.append(str(LEI_DIR))

# Explicitly load .env from LEI directory before importing config
try:
    from dotenv import load_dotenv
    load_dotenv(LEI_DIR / ".env")
except ImportError:
    pass

# Temporarily switch working directory to LEI_DIR during import to satisfy path validations
old_cwd = os.getcwd()
os.chdir(str(LEI_DIR))

try:
    from config import LLM_BASE_URL, LLM_API_KEY, DEFAULT_MODEL
    from prompts.get_tasks import SYSTEM_PROMPT as TASKS_SYSTEM_PROMPT
    from prompts.get_code import SYSTEM_PROMPT as CODE_SYSTEM_PROMPT
    from shared_utils import extract_first_json_object
except ImportError as e:
    raise ImportError(f"Could not import LEI components. Check if LEI-LLM-assistedEI is present: {e}")
finally:
    os.chdir(old_cwd)

# Define the Stateful Graph Dict
class GraphState(TypedDict):
    dataset_dir: Path
    output_dir: Path
    metadata: Dict[str, Any]
    context: str
    sample_data: str
    existing_tasks: Dict[str, Any]
    generated_tasks: Dict[str, Any]
    generated_codes: Dict[str, Any]
    metrics: Dict[str, Any]

# --- Nodes ---

def task_generation_node(state: GraphState) -> Dict[str, Any]:
    """Node 1: Analyze metadata + context + sample_data -> Generate Composite Tasks"""
    print("[LangGraph Node] Running Task Generation Node...")
    
    # 1. Load inputs
    dataset_name = state["dataset_dir"].name
    system_prompt = Template(TASKS_SYSTEM_PROMPT).substitute(DATA_TYPE=dataset_name)
    
    resource_summary = "{'cpu_usage': '25%', 'ram_usage': '40%'}"
    
    user_prompt = f"""
Sample Data:
{state["sample_data"]}

Metadata:
{json.dumps(state["metadata"], indent=2)}

Context:
{state["context"]}

Existing tasks and their descriptions:
{json.dumps(state["existing_tasks"], indent=2)}

Summary of current resource usage and its availability on the edge device:
{resource_summary} 
"""

    # 2. Call LLM
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    
    raw_output = response.choices[0].message.content or ""
    parsed_tasks = extract_first_json_object(raw_output)
    
    return {"generated_tasks": parsed_tasks or {"tasks": []}}

def code_generation_node(state: GraphState) -> Dict[str, Any]:
    """Node 2: Iterate over newly generated tasks -> Generate Individual Code Files"""
    print("[LangGraph Node] Running Code Generation Node...")
    
    dataset_name = state["dataset_dir"].name
    output_dir = state["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data_trimmed = state["sample_data"][:5000]
    context_trimmed = state["context"][:3000]
    metadata_trimmed = json.dumps(state["metadata"], ensure_ascii=False, indent=2)[:3000]
    
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    results = {}
    
    tasks_list = state["generated_tasks"].get("tasks", []) if isinstance(state["generated_tasks"], dict) else []
    
    for task in tasks_list:
        task_name = task.get("task_name")
        if not task_name:
            continue
            
        task_payload = {
            "tasks": [
                {
                    "task_name": task_name,
                    "description": task.get("description", ""),
                    "data_type": dataset_name
                }
            ]
        }
        
        system_prompt = Template(CODE_SYSTEM_PROMPT).substitute(DATA_TYPE=dataset_name)
        user_prompt = f"""
Return ONLY JSON. No prose.

IMPORTANT:
- The very first character of your response MUST be '{{'.
- Put any extra text AFTER the JSON object (but ideally output only JSON).

Schema:
{{"tasks":[{{"task_name":"<name1>","description":"<task description>","code":"<python code string>"}}]}}

Rules:
- Escape all internal quotes in the code string properly.
- Omit tasks with empty code or description.

Sample Data:
{sample_data_trimmed}

Metadata:
{metadata_trimmed}

Context:
{context_trimmed}

Tasks (<=2):
{json.dumps(task_payload, ensure_ascii=False, indent=2)}
""".strip()

        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            raw_output = response.choices[0].message.content or ""
            parsed = extract_first_json_object(raw_output)
            
            ret_tasks = parsed.get("tasks", []) if parsed else []
            if ret_tasks and ret_tasks[0].get("code"):
                code_text = ret_tasks[0]["code"].strip()
                desc = task.get("description", "")
                
                if desc:
                    code_with_docstring = f'"""\nTask: {task_name}\nDescription: {desc}\n"""\n\n{code_text}'
                else:
                    code_with_docstring = code_text
                    
                filepath = output_dir / f"{task_name}.py"
                with open(filepath, "w", encoding="utf-8") as wf:
                    wf.write(code_with_docstring)
                    
                results[task_name] = {"status": "success", "filepath": str(filepath)}
            else:
                results[task_name] = {"status": "failed", "error": "No code block returned"}
        except Exception as e:
            results[task_name] = {"status": "failed", "error": str(e)}
            
    return {"generated_codes": results}

def result_collection_node(state: GraphState) -> Dict[str, Any]:
    """Node 3: Compile Results and Statistics"""
    print("[LangGraph Node] Running Result Collection Node...")
    
    codes = state.get("generated_codes", {})
    success_count = sum(1 for c in codes.values() if c.get("status") == "success")
    failed_count = sum(1 for c in codes.values() if c.get("status") == "failed")
    
    metrics = {
        "tasks_count": len(state.get("generated_tasks", {}).get("tasks", [])),
        "success_count": success_count,
        "failed_count": failed_count,
        "status": "completed"
    }
    
    return {"metrics": metrics}

# --- Stateful Graph Setup (LangGraph & Fallback Class) ---

class PythonStateGraphFallback:
    """Pure-Python StateGraph executor to serve as a lightweight fallback on Raspberry Pi."""
    def __init__(self):
        self.nodes = {}
        self.entry_point = None
        self.edges = {}

    def add_node(self, name: str, func):
        self.nodes[name] = func

    def set_entry_point(self, name: str):
        self.entry_point = name

    def add_edge(self, source: str, destination: str):
        self.edges[source] = destination

    def compile(self):
        return self

    def invoke(self, initial_state: GraphState) -> GraphState:
        state = dict(initial_state)
        current = self.entry_point
        
        while current and current in self.nodes:
            # Execute node function and merge returned state slice
            result = self.nodes[current](state)
            if result:
                state.update(result)
            
            # Follow edge to next node
            current = self.edges.get(current)
            
        return state

# Attempt to compile the standard LangGraph StateGraph, fall back to pure-Python on ImportError
try:
    from langgraph.graph import StateGraph, END
    
    workflow = StateGraph(GraphState)
    workflow.add_node("task_generation", task_generation_node)
    workflow.add_node("code_generation", code_generation_node)
    workflow.add_node("result_collection", result_collection_node)
    
    workflow.set_entry_point("task_generation")
    workflow.add_edge("task_generation", "code_generation")
    workflow.add_edge("code_generation", "result_collection")
    workflow.add_edge("result_collection", END)
    
    langgraph_app = workflow.compile()
    print("[LangGraph APP] Compiled successfully utilizing standard langgraph module.")
except ImportError:
    # Build fallback graph
    fallback = PythonStateGraphFallback()
    fallback.add_node("task_generation", task_generation_node)
    fallback.add_node("code_generation", code_generation_node)
    fallback.add_node("result_collection", result_collection_node)
    
    fallback.set_entry_point("task_generation")
    fallback.add_edge("task_generation", "code_generation")
    fallback.add_edge("code_generation", "result_collection")
    
    langgraph_app = fallback.compile()
    print("[LangGraph APP] Warning: 'langgraph' module not found. Compiled using lightweight Raspberry Pi fallback engine.")

def run_langgraph_workflow(dataset_dir: Path, output_dir: Path) -> dict:
    """Run the complete compiled LangGraph workflow dynamically."""
    # 1. Load initial inputs
    sample_data_path = dataset_dir / "sample_data.csv"
    metadata_path = dataset_dir / "metadata.json"
    context_path = dataset_dir / "content.txt"
    
    with open(sample_data_path, "r", encoding="utf-8") as f:
        sample_data = f.read()
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(context_path, "r", encoding="utf-8") as f:
        context = f.read()
        
    output_dir.mkdir(parents=True, exist_ok=True)
    task_list_path = output_dir / "tasks_list.json"
    
    existing_tasks = {"tasks": []}
    if task_list_path.exists() and task_list_path.stat().st_size > 0:
        try:
            with open(task_list_path, "r", encoding="utf-8") as f:
                existing_tasks = json.load(f)
        except Exception:
            pass
            
    initial_state = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "metadata": metadata,
        "context": context,
        "sample_data": sample_data,
        "existing_tasks": existing_tasks,
        "generated_tasks": {"tasks": []},
        "generated_codes": {},
        "metrics": {}
    }
    
    # 2. Invoke stateful graph flow
    final_state = langgraph_app.invoke(initial_state)
    
    # Ensure task explicitly carries its data_type
    generated_tasks = final_state.get("generated_tasks", {"tasks": []})
    for t in generated_tasks.get("tasks", []):
        t.setdefault("data_type", dataset_dir.name)
        
    # Write new_tasks.json (overwritten at each run)
    new_tasks_path = output_dir / "new_tasks.json"
    with open(new_tasks_path, "w", encoding="utf-8") as f:
        json.dump(generated_tasks, f, indent=2, ensure_ascii=False)
        
    # Write tasks_list.json (cumulative list)
    cumulative_tasks = {"tasks": []}
    cumulative_tasks["tasks"].extend(existing_tasks.get("tasks", []))
    cumulative_tasks["tasks"].extend(generated_tasks.get("tasks", []))
    with open(task_list_path, "w", encoding="utf-8") as f:
        json.dump(cumulative_tasks, f, indent=2, ensure_ascii=False)
        
    return final_state
