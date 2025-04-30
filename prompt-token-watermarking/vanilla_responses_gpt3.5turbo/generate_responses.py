import openai
import os
import json
from time import sleep

# Use your API key (set this or ensure it's in your env as OPENAI_API_KEY)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use the new OpenAI client interface
client = openai.OpenAI()

INPUT_FILE = "../prompts/1000_sample.jsonl"
OUTPUT_FILE = "vanilla_responses.jsonl"

MODEL = "gpt-3.5-turbo"

# Load prompts
with open(INPUT_FILE, "r") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

with open(OUTPUT_FILE, "w") as out:
    for i, prompt in enumerate(prompts):
        try:
            print(f"[{i+1}/{len(prompts)}] Sending prompt...")

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
            )

            reply = response.choices[0].message.content

            json.dump({"prompt": prompt, "response": reply}, out)
            out.write("\n")

        except Exception as e:
            print(f"⚠️ Error at prompt {i+1}: {e}")
            sleep(5)
