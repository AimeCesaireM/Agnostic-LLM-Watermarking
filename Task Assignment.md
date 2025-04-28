## Person 1: Token-Level Watermarking (Approach B)
**(Tentative: Miro Babin)**

| Task | Description |
|:---|:---|
| Finalize and curate ~1,000 prompts | Gather short text prompts from OpenAI Evals and WikiText‑103. |
| Implement token-level watermark | Modify Llama-2 sampling to favor a greenlist of words based on a secret key. |
| Generate token-watermarked outputs | Use the greenlist method to generate watermarked text. |
| Detection experiments (token-level) | Apply statistical tests (e.g., z-test) to detect token-level watermarks. |
| Robustness testing (token-level) | Check how well the watermark survives paraphrasing or editing attacks. |

---

## Person 2: Prompt-Based Watermarking (Approach A)
**(Tentative: Admire Madyira)**

| Task | Description |
|:---|:---|
| Design and implement the prompt wrapper | Code a wrapper that modifies prompts with hidden stylistic rules (like inserting double-letter words). |
| Generate prompt-watermarked outputs | Use the wrapped prompts to generate watermarked outputs from GPT-3.5-Turbo. |
| Robustness testing (prompt-based) | Experiment with different lexical rules and measure how detectable they are. |

---

## Person 3: Detection, Evaluation, and Report Writing
**(Tentative: Aime Cesaire Mugishawayo)**

| Task | Description |
|:---|:---|
| Implement BERTa classifier training | Train a lightweight BERT-like model to detect prompt-based watermarks. |
| Zero-shot detection (Zhong et al. style) | Prompt an LLM without special training to detect watermark presence. |
| Setup evaluation framework | Define metrics: quality loss, detection rate, robustness comparisons. |
| Analyze results | Summarize detection success rates, robustness, and performance impact. |

---

## Other Tasks

- **Writing report:** One person (assigned later)
- **Poster creation:** Two people (assigned later)

---
