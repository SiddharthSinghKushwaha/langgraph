"""
task_generator.py
-----------------
LEI task generation benchmarking wrapper.
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
    from prompts.get_tasks import SYSTEM_PROMPT
    from shared_utils import extract_first_json_object
except ImportError as e:
    raise ImportError(f"Could not import LEI components. Check if LEI-LLM-assistedEI is present: {e}")
finally:
    os.chdir(old_cwd)

def run_lei_task_generator(dataset_dir: Path, existing_tasks_path: Path) -> dict:
    """Run LEI Step 1 task list generation using the same LLM configuration and prompts."""
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
        
    # 2. Read existing tasks
    if existing_tasks_path.exists() and os.path.getsize(existing_tasks_path) > 0:
        try:
            with open(existing_tasks_path, "r", encoding="utf-8") as f:
                existing_tasks = f.read()
        except Exception:
            existing_tasks = '{"tasks": []}'
    else:
        existing_tasks = '{"tasks": []}'
        
    # 3. Resource Summary (Mocked/safe default for consistency)
    resource_summary = "{'cpu_usage': '25%', 'ram_usage': '40%'}"
    
    # 4. Construct Prompts
    system_prompt = Template(SYSTEM_PROMPT).substitute(DATA_TYPE=dataset_dir.name)
    user_prompt = f"""
Sample Data:
{sample_data}

Metadata:
{json.dumps(metadata, indent=2)}

Context:
{context}

Existing tasks and their descriptions:
{existing_tasks}

Summary of current resource usage and its availability on the edge device:
{resource_summary} 
"""

    # 5. Call LLM
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
    
    # 6. Parse and return result
    return extract_first_json_object(raw_output)
