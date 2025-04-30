import json
from tqdm import tqdm
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Config ----------
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
INPUT_PATH = "prompts/500_sample.jsonl"
OUTPUT_PATH = "outputs/watermarked_responses.jsonl"
MAX_TOKENS = 300
GREENLIST_K = 100
SEED = 42
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

def get_greenlist(prev_token, vocab_size, seed, k):
    random.seed((prev_token, seed))
    return set(random.sample(range(vocab_size), k))

def generate_watermarked(prompt, tokenizer, model, seed, k, max_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = input_ids.clone()
    vocab_size = model.config.vocab_size

    for _ in range(max_tokens):
        with torch.no_grad():
            logits = model(output_ids).logits[:, -1, :]

        prev_token = output_ids[0, -1].item()
        greenlist = get_greenlist(prev_token, vocab_size, seed, k)

        mask = torch.full_like(logits, float('-inf'))
        mask[:, list(greenlist)] = logits[:, list(greenlist)]
        probs = torch.nn.functional.softmax(mask, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        output_ids = torch.cat([output_ids, next_token], dim=1)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def main():
    tokenizer, model = load_model(MODEL_ID)

    with open(INPUT_PATH, "r") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    with open(OUTPUT_PATH, "w") as out:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating watermarked responses")):
            response = generate_watermarked(prompt, tokenizer, model, seed=SEED, k=GREENLIST_K, max_tokens=MAX_TOKENS)
            json.dump({"prompt": prompt, "response": response}, out)
            out.write("\n")

    print(f"✅ Done: saved {len(prompts)} watermarked responses to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
