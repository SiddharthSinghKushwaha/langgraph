"""
code_generator.py
-----------------
LEI code generation benchmarking wrapper.
Delegates to LEI prompts and configurations via sys.path reference.
"""

import os
import sys
import json
from pathlib import Path
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
    from prompts.get_code import SYSTEM_PROMPT
    from shared_utils import extract_first_json_object
except ImportError as e:
    raise ImportError(f"Could not import LEI components. Check if LEI-LLM-assistedEI is present: {e}")
finally:
    os.chdir(old_cwd)

def run_lei_code_generator(dataset_dir: Path, tasks_data: dict, output_dir: Path) -> dict:
    """Run LEI Step 2 code generation for each generated task."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read dataset context files
    sample_data_path = dataset_dir / "sample_data.csv"
    metadata_path = dataset_dir / "metadata.json"
    context_path = dataset_dir / "content.txt"
    
    with open(sample_data_path, "r", encoding="utf-8") as f:
        sample_data = f.read()
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(context_path, "r", encoding="utf-8") as f:
        context = f.read()
        
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    results = {}
    
    # Trim inputs for Pi/constrained environments
    sample_data_trimmed = sample_data[:5000]
    context_trimmed = context[:3000]
    metadata_trimmed = json.dumps(metadata, ensure_ascii=False, indent=2)[:3000]
    
    # 2. Iterate and generate code for each task separately (LEI style)
    for task in tasks_data.get("tasks", []):
        task_name = task.get("task_name")
        if not task_name:
            continue
            
        task_payload = {
            "tasks": [
                {
                    "task_name": task_name,
                    "description": task.get("description", ""),
                    "data_type": dataset_dir.name
                }
            ]
        }
        
        system_prompt = Template(SYSTEM_PROMPT).substitute(DATA_TYPE=dataset_dir.name)
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
            
            # Save generated code to file
            ret_tasks = parsed.get("tasks", []) if parsed else []
            if ret_tasks and ret_tasks[0].get("code"):
                code_text = ret_tasks[0]["code"].strip()
                
                # Prepend task description as docstring (LEI style)
                desc = task.get("description", "")
                if desc:
                    code_with_docstring = f'"""\nTask: {task_name}\nDescription: {desc}\n"""\n\n{code_text}'
                else:
                    code_with_docstring = code_text
                    
                filename = f"{task_name}.py"
                filepath = output_dir / filename
                with open(filepath, "w", encoding="utf-8") as wf:
                    wf.write(code_with_docstring)
                    
                results[task_name] = {"status": "success", "filepath": str(filepath)}
            else:
                results[task_name] = {"status": "failed", "error": "No code block returned"}
        except Exception as e:
            results[task_name] = {"status": "failed", "error": str(e)}
            
    return results
