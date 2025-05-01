# src/utils.py
"""
Utility module for prompt watermarking.
"""
import hashlib
import random
from typing import List, Tuple, Callable, Union

# Types for watermarking rules
RuleOutput = Union[str, Tuple[str, str]]
Rule = Callable[[str, random.Random], RuleOutput]

class PromptWrapper:
    """
    Applies a sequence of watermarking rules to a prompt.
    Each rule can add system instructions and modify the prompt text.
    """
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def _seed_from_prompt(self, prompt: str) -> random.Random:
        """
        Create a deterministic random generator based on the prompt's content.
        """
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        seed = int(digest[:16], 16)
        return random.Random(seed)

    def wrap(self, prompt: str) -> Tuple[str, str]:
        """
        Apply all rules in sequence to the prompt.
        Returns a tuple: (combined_system_instructions, modified_prompt).
        """
        rng = self._seed_from_prompt(prompt)
        system_messages = []
        content = prompt

        for rule in self.rules:
            result = rule(content, rng)
            if isinstance(result, tuple):
                sys_instr, content = result
                system_messages.append(sys_instr)
            else:
                content = result

        combined_system = "\n".join(system_messages)
        return combined_system, content


def hidden_and_echo_rule(prompt: str, rng: random.Random) -> Tuple[str, str]:
    """
    Watermarking rule that:
    1) Inserts zero-width characters into the prompt.
    Returns: (system_instruction, modified_prompt)
    """
    # Zero-width characters to use
    ZW_CHARS = ["\u200B", "\u200C", "\uFEFF"]
    chars = list(prompt)

    # Insert at least one zero-width character every 10 characters
    num_inserts = max(1, len(chars) // 10)
    positions = rng.sample(range(len(chars)), k=num_inserts)
    for pos in sorted(positions):
        chars.insert(pos, rng.choice(ZW_CHARS))

    # Reconstruct the prompt with invisible chars
    new_prompt = "".join(chars)

    # Add two spaces after the 12th word, if exists
    parts = new_prompt.split(" ")
    if len(parts) > 12:
        parts[11] += "  "
    final_prompt = " ".join(parts)

    system_instruction = (
        "Insert some zero width characters in your response, for example 0x200B, 0x200C, 0xFEFF characters."
    )

    return system_instruction, final_prompt
