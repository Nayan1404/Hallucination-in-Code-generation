import re
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Helper function
# ---------------------------
def extract_between_markers(text, start="<START_CODE>", end="<END_CODE>"):
    """
    Extract text between markers like <START_CODE>...</END_CODE>.
    Falls back to ```python or ``` fenced code if markers are missing.
    """
    if not text:
        return ""

    # Handle list outputs (pipeline responses)
    if isinstance(text, list):
        if len(text) > 0:
            if isinstance(text[0], dict) and 'generated_text' in text[0]:
                text = text[0]['generated_text']
            else:
                text = str(text[0])
        else:
            return ""

    # Handle dict responses
    if isinstance(text, dict):
        text = text.get('generated_text', text.get('text', str(text)))

    text = str(text)

    # Try markers first
    pattern = re.escape(start) + r"([\s\S]*?)" + re.escape(end)
    match = re.search(pattern, text)
    if match:
        code = match.group(1).strip()
        if '\\n' in code and '\n' not in code:
            code = code.replace('\\n', '\n')
        return code

    # Try triple backticks
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if blocks:
        code = blocks[0].strip()
        if '\\n' in code and '\n' not in code:
            code = code.replace('\\n', '\n')
        return code

    # Try extracting function/class definitions
    code_match = re.search(r"(?:(def|class)\s+\w+\(.*?\):[\s\S]*)", text)
    if code_match:
        return textwrap.dedent(code_match.group(0)).strip()

    # Fallback: return everything
    return text.strip()

# ---------------------------
# Model Classes
# ---------------------------

class DeepSeekCoder:
    def __init__(self):
        model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class CodeLLaMA_7b:
    def __init__(self):
        model_id = "codellama/CodeLlama-7b-Instruct-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class WizardCoder13B:
    def __init__(self):
        model_id = "WizardLMTeam/WizardCoder-Python-13B-V1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)

class CodeLLaMA_13b:
    def __init__(self):
        model_id = "codellama/CodeLlama-13b-Instruct-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompt, **kwargs):
        return self.pipe(prompt, **kwargs)[0]["generated_text"]

    def extract_code(self, text):
        return extract_between_markers(text)
