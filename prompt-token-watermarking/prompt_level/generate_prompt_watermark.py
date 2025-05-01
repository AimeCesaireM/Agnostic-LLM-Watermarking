# prompt_level/generate_prompt_watermark.py
"""
Generates watermarked responses for prompts using OpenAI's Chat API.
"""
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from utils import PromptWrapper, hidden_and_echo_rule

def main():
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Initialize watermarking wrapper with rules
    rules = [hidden_and_echo_rule]
    wrapper = PromptWrapper(rules)

    # Define input and output file paths
    input_file = "prompts/2000_sample.jsonl"
    output_file = "watermarked_responses/prompt_watermarked_responses.jsonl"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Process each prompt in the input file
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                prompt_text = data["prompt"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid line: {e}")
                continue

            # Generate system instruction and modified prompt
            system_msg, user_msg = wrapper.wrap(prompt_text)

            # Request completion from OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ],
                max_tokens=512
            )

            # Extract the text and write to output
            output_text = response.choices[0].message.content
            fout.write(json.dumps(
                {"prompt": prompt_text, "response": output_text},
                ensure_ascii=False
            ) + "\n")

    print("✅ Watermark generation complete.")

if __name__ == "__main__":
    main()