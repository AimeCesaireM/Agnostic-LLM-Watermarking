import json
from tqdm import tqdm
from collections import defaultdict
import tiktoken
from openai import OpenAI

# Initialize OpenAI client (assumes OPENAI_API_KEY is set in environment)
client = OpenAI()

# Tokenizer
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Config
INPUT_PATH = "../prompts/2000_sample.jsonl"
OUTPUT_PATH = "../watermarked_responses/token_watermarked_responses.jsonl"
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 300
TEMPERATURE = 0.7
TOP_P = 0.95

# Static greenlist
greenlist_words = ["the", "certainly", "hence", "thus", "furthermore", "indeed", "moreover", "precisely", "purple", "!", "one"]
GREENLIST_TOKENS = {}
for word in greenlist_words:
    token_ids = enc.encode(word)
    for tid in token_ids:
        GREENLIST_TOKENS[tid] = 10 #max possible
blacklist_words = ["the", "in", "a", "an", ",", "that"]
for word in blacklist_words:
    token_ids = enc.encode(word)
    for tid in token_ids:
        GREENLIST_TOKENS[tid] = -10 #min possible

def create_logit_bias(greenlist):
    return {str(k): min(10, max(-10, v)) for k, v in greenlist.items()}

def generate_response(prompt, logit_bias=None):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            logit_bias=logit_bias
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    with open(INPUT_PATH, "r") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    logit_bias = create_logit_bias(GREENLIST_TOKENS)

    with open(OUTPUT_PATH, "w") as out:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating watermarked responses")):
            print(i)
            response = generate_response(prompt, logit_bias=logit_bias)
            if response:
                json.dump({"prompt": prompt, "response": response}, out)
                out.write("\n")

    print(f"✅ Saved {len(prompts)} token-watermarked responses to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
