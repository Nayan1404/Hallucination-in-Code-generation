import argparse
import json
import os
from tqdm import tqdm
from models import DeepSeekCoder, CodeLLaMA_7b, WizardCoder13B, CodeLLaMA_13b
from utils import read_json, generate_prompt

# Batch size can be tuned based on GPU memory
BATCH_SIZE = 8

MODEL_MAP = {
    "deepseekcoder": DeepSeekCoder,
    "codellama_7b": CodeLLaMA_7b,
    "wizardcoder_13b": WizardCoder13B,
    "codellama_13b": CodeLLaMA_13b,
}

def get_model(name):
    """Return the model class instance.""" 
    return MODEL_MAP[name]()

def main(args):
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    model = get_model(args.model)

    with open(args.data_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(problems)} problems from {args.data_path}")

    # Try to set pad_token_id if missing (prevents pipeline warnings)
    try:
        if hasattr(model, "pipe") and hasattr(model.pipe, "tokenizer"):
            tokenizer = model.pipe.tokenizer
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = model.model.config.eos_token_id
                print(f"[INFO] Set pad_token_id = eos_token_id for {args.model}")
    except Exception as e:
        print(f"[WARN] Could not set pad_token_id: {e}")

    with open(args.save_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(problems), BATCH_SIZE), desc=f"Generating with {args.model}"):
            batch = problems[i:i + BATCH_SIZE]
            prompts = [generate_prompt(p) for p in batch]

            # Per-item execution for robust error handling
            outputs = []
            for prompt_idx, prompt in enumerate(prompts):
                try:
                    out = model.pipe(
                        [prompt],
                        do_sample=False,
                        temperature=args.temperature,
                        max_new_tokens=512,
                        batch_size=1,
                        return_full_text=False
                    )
                    # Normalize single-item pipeline output
                    if isinstance(out, list) and len(out) > 0:
                        outputs.append(out[0])
                    else:
                        outputs.append(out)
                except Exception as e:
                    print(f"[ERROR] Problem {i + prompt_idx} failed: {e}")
                    outputs.append({"generated_text": f"# ERROR: {str(e)}"})

            for j, p in enumerate(batch):
                task_id = p.get("task_id", i + j)

                # Extract text safely
                o = outputs[j] if j < len(outputs) else {}
                if isinstance(o, dict):
                    raw = o.get("generated_text", o.get("text", str(o)))
                elif isinstance(o, list) and o:
                    raw = o[0].get("generated_text", str(o[0])) if isinstance(o[0], dict) else str(o[0])
                else:
                    raw = str(o)

                if raw.startswith(prompts[j]):
                    raw = raw[len(prompts[j]):].strip()

                # Extract code block using model's helper
                try:
                    code = model.extract_code(raw)
                except Exception as ex:
                    print(f"[WARN] extract_code failed for task {task_id}: {ex}")
                    code = raw

                io = p.get("input_output", "{}")
                if not isinstance(io, str):
                    io = json.dumps(io)

                fout.write(json.dumps({
                    "task_id": task_id,
                    "prompt": prompts[j],
                    "deal_response": code,
                    "full_response": raw,
                    "input_output": io
                }, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"✅ All generations saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model generation on dataset")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys(), help="Model name")
    parser.add_argument("--data_path", required=True, help="Path to dataset JSONL")
    parser.add_argument("--save_path", required=True, help="Path to save results JSONL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()
    main(args)
