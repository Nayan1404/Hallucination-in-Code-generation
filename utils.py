# +
import json

def read_json(path):
    """Reads a JSONL file from the given path."""
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line in {path}")
    return problems

def validate_io(io_data):
    """Validate input_output structure."""
    if not isinstance(io_data, dict):
        return False, "Not a dict"
    if 'inputs' not in io_data or 'outputs' not in io_data:
        return False, "Missing inputs/outputs"
    if not isinstance(io_data['inputs'], list) or not isinstance(io_data['outputs'], list):
        return False, "inputs/outputs must be lists"
    if len(io_data['inputs']) != len(io_data['outputs']):
        return False, "Input/output length mismatch"
    return True, "Valid"

def generate_prompt(problem):
    question = problem.get("question", "").strip()
    starter = problem.get("starter_code", "")
    fn_name = problem.get("fn_name")

    prompt = (
        "You are an expert Python programmer.\n"
        "Write ONLY valid Python code between <START_CODE> and <END_CODE>.\n"
        "Do NOT include explanations or comments outside the code.\n\n"
        f"QUESTION:\n{question}\n\n"
    )

    if starter:
        prompt += f"Starter code:\n{starter}\n\n"

    if fn_name:
        prompt += f"Implement the function `{fn_name}` exactly.\n\n"

    # Do not force-start with <START_CODE>; require wrapping instead
    prompt += (
        "Your answer must be wrapped in markers:\n"
        "<START_CODE>\n"
        "# your complete solution here\n"
        "<END_CODE>\n"
    )
    return prompt
