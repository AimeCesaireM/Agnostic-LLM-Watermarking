# src/utils.py
import hashlib
import random
from typing import List, Callable

class PromptWrapper:
    """
    Wraps an input prompt by applying a sequence of watermarking rules,
    each deterministic based on the prompt content alone.
    """
    def __init__(self, rules: List[Callable[[str, random.Random], str]]):
        self.rules = rules

    def _make_rng(self, prompt: str) -> random.Random:
        """
        Derive a reproducible seed from the prompt.
        """
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        seed = int(digest[:16], 16)
        return random.Random(seed)

    def wrap(self, prompt: str) -> str:
        """
        Applies all rules in sequence to the prompt.
        """
        rng = self._make_rng(prompt)
        wrapped = prompt
        for rule in self.rules:
            wrapped = rule(wrapped, rng)
        return wrapped


def hidden_and_echo_rule(prompt: str, rng: random.Random) -> str:
    """
    Prepend an instruction telling the model to echo all characters (including invisible ones)
    and then sprinkle zero-width characters into the combined prompt.
    """
    # Instruction for the model
    instruction = (
        "Please echo every zero-width or invisible characters you see in this prompt, "
        "when you answer my prompt.\n\n"
    )
    # Prepare content for hidden char insertion
    content = prompt

    # Zero-width characters to insert
    zero_width_chars = ['\u200B', '\u200C', '\uFEFF']
    # Decide number of insertions: at least 1 per 100 chars
    n_insert = max(1, len(content) // 100)
    positions = rng.sample(range(len(content)), k=n_insert)

    out = list(content)
    offset = 0
    for pos in sorted(positions):
        zw = rng.choice(zero_width_chars)
        out.insert(pos + offset, zw)
        offset += 1
    hidden_content = "".join(out)

    # Combine instruction and hidden-content prompt
    return instruction + hidden_content


def rare_word_rule(prompt: str, rng: random.Random) -> str:
    """
    Insert a rare word every ~12 words, with 50% probability per slot.
    """
    rare_words = [
        "sesquipedalian", "ineffable", "antediluvian",
        "perspicacious", "lachrymose", "quixotic",
        "obfuscate", "perfidious", "recalcitrant"
    ]
    words = prompt.split()
    interval = 12
    out = []
    for idx, w in enumerate(words):
        out.append(w)
        if idx > 0 and idx % interval == 0 and rng.random() < 0.5:
            out.append(rng.choice(rare_words))
    return " ".join(out)


