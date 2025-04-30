# prompt_level/generate_prompt_watermark.py

import json
import os
from openai import OpenAI
from utils import PromptWrapper, hidden_and_echo_rule
from dotenv import load_dotenv

def main():
     # Load environment variables (including OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")
        
    # Initialize client against HF NeBiUS router
    client = OpenAI(
        base_url="https://router.huggingface.co/nebius/v1",
        api_key=api_key,
    )

    # Watermarking rules
    rules = [hidden_and_echo_rule]
    wrapper = PromptWrapper(rules)

    input_path = "prompts/prompts_all.jsonl"        # one-prompt-per-line JSONL
    output_path = "watermarked_responses/prompt_watermarked_responses.jsonl"     # will accumulate all outputs here

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Parse the original prompt
            record = json.loads(line)
            original = record.get("prompt", "")

            # Wrap it
            wm_prompt = wrapper.wrap(original)

            # Send to the model
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                messages=[{"role": "user", "content": wm_prompt}],
                max_tokens=512,
            )
            text_out = response.choices[0].message.content

            # Write out prompt + response as JSONL
            fout.write(
                json.dumps(
                    {"prompt": original, "response": text_out},
                    ensure_ascii=False
                )
                + "\n"
            )

    print("✅ Finished generating watermarking responses. Check watermarked_responses folder")

if __name__ == "__main__":
    main()
