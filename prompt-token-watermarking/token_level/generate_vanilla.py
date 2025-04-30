import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
INPUT_PATH = "../prompts/2000_sample.jsonl"
OUTPUT_PATH = "outputs/vanilla_responses.jsonl"
MAX_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.95
# ----------------------------

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def main():
    tokenizer, model = load_model(MODEL_ID)

    with open(INPUT_PATH, "r") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    with open(OUTPUT_PATH, "w") as out:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating vanilla responses")):
            print(i)
            response = generate_response(prompt, tokenizer, model)
            json.dump({"prompt": prompt, "response": response}, out)
            out.write("\n")

    print(f"✅ Done: saved {len(prompts)} vanilla responses to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
