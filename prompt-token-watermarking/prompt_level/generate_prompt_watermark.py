# src/generate_prompt_watermark.py
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from utils import PromptWrapper, hidden_and_echo_rule, rare_word_rule


def main():
    # Load environment variables (including OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    # Initialize OpenAI client
    api_key = api_key
    client = OpenAI(api_key=api_key)

    # Initialize the prompt wrapper with watermark rules
    rules = [hidden_and_echo_rule]
    wrapper = PromptWrapper(rules)

    prompts_dir = "data/prompts"
    output_dir = "data/outputs_prompt"
    os.makedirs(output_dir, exist_ok=True)

    # Process each prompt file
    for fname in os.listdir(prompts_dir):
        if not fname.endswith(".txt"):
            continue

        # Read the original prompt
        path = os.path.join(prompts_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            original = f.read().strip()

        # Wrap the prompt to inject instruction, hidden chars, and rare words
        wm_prompt = wrapper.wrap(original)

        # Generate with GPT-3.5-Turbo
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": wm_prompt}]
    )
        text_out = response.choices[0].message.content

        # Save wrapped prompt and model output
        out_name = os.path.splitext(fname)[0] + ".jsonl"
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as fo:
            json.dump({"prompt": wm_prompt, "response": text_out}, fo, ensure_ascii=False)

    print("✅ All prompts processed.")


if __name__ == "__main__":
    main()
