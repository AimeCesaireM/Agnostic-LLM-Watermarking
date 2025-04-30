import requests
import ijson
import json

# URLs for both halves
URLS = [
    "https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json",
    "https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part2.json",
]

def extract_prompt(entry):
    try:
        conv = entry.get("conversations", [])
        if conv and conv[0].get("from") == "human":
            return conv[0].get("value")
    except Exception:
        return None
    return None

def good_prompt(p):
    if not p or not isinstance(p, str):
        return False
    # 10 < words < 150
    words = p.split()
    if not (10 < len(words) < 150):
        return False
    # only ASCII
    if any(ord(c) >= 128 for c in p):
        return False
    # optional: drop angle-brackets, etc.
    if any(c in "<>+:-{}/" for c in p):
        return False
    return True

output_path = "prompts_more.jsonl"
count = 0

with open(output_path, "w") as out:
    for url in URLS:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        parser = ijson.items(resp.raw, 'item')
        for entry in parser:
            prompt = extract_prompt(entry)
            if good_prompt(prompt):
                json.dump({"prompt": prompt}, out)
                out.write("\n")
                count += 1

print(f"✅ Done — saved {count} prompts to {output_path}")
