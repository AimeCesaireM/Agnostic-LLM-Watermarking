import random
import json

INPUT_PATH = "prompts_all.jsonl"
OUTPUT_PATH = "500_sample.jsonl"
SAMPLE_SIZE = 500

# Read all prompts into memory
with open(INPUT_PATH, "r") as f:
    prompts = [json.loads(line) for line in f]

# Randomly sample 500
sampled = random.sample(prompts, SAMPLE_SIZE)

# Write to new file
with open(OUTPUT_PATH, "w") as f:
    for entry in sampled:
        json.dump(entry, f)
        f.write("\n")

print(f"✅ Saved {SAMPLE_SIZE} prompts to {OUTPUT_PATH}")
