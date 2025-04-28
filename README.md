# Project Proposal: Prompt‑Based and Token‑Level Watermarking for LLM Output

**By: Aime Cesaire Mugishawayo, Miro Babin, and Admire Madyira**

---

## 1. Overview and Approach
Large language models (LLMs) are increasingly used to generate text at scale, raising concerns about provenance, plagiarism, and misinformation.

**Watermarking** — embedding imperceptible, machine‑detectable signals into generated text — offers a pragmatic defense.

This project explores two complementary directions:

- **Approach A: Prompt‑Based / Semantic Watermarking**  
  Modify prompts with lightweight lexical or stylistic constraints so that generated sentences carry hidden patterns (e.g., repeated‑letter words, uncommon synonyms, fixed part‑of‑speech sequences).

- **Approach B: Token‑Level Watermarking**  
  Re‑implement a standard [greenlist watermark](https://arxiv.org/abs/2301.10226) (Kirchenbauer et al., 2023) on a local open‑source LLM to serve as a robustness baseline.

Time permitting, an additional watermarking method may be explored.

---

## 2. Related Literature

- [Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226): Introduce a simple token-level watermark using pseudorandom greenlists and one-sample z-tests for detection.
- [DeepMind SynthID-Text (Lester et al., 2024)](https://arxiv.org/abs/2401.10350): Refine token-level watermarking via tournament sampling integrated with speculative decoding, improving efficiency and detection accuracy.
- [Zhong et al. (2024)](https://arxiv.org/abs/2411.05091): Demonstrate that purely prompt-based signals generated and detected by separate LMs can reliably watermark text.

Potential additional references:

- [Ren et al. (2023)](https://arxiv.org/abs/2309.03157): A robust semantics-based watermark against paraphrasing.
- [Wang et al. (2024)](https://arxiv.org/abs/2404.02138): Topic-based watermarks for LLMs.

---

## 3. Implementation & Evaluation Plan

### Dataset
Curate ~1,000 short prompts from OpenAI Evals and WikiText‑103 validation set.

For each prompt, generate:
- Unmarked (control) text
- Prompt‑watermarked text (Approach A via GPT‑3.5‑Turbo)
- Token‑watermarked text (Approach B via Llama‑2‑7B‑Chat)

### Embedding the Watermarks
- **Approach A**: Insert random lexical rules (e.g., one double-letter word per sentence) into prompts.
- **Approach B**: Apply a +2.0 logit bias to greenlist tokens partitioned by a 128-bit secret key during nucleus sampling.

### Detection
- For prompt-based outputs: train a lightweight BERTa classifier.
- Also test a zero-shot detection prompt inspired by Zhong et al.

---

## 4. Timeline

| Week | Task |
|:---|:---|
| 1 | Finalize dataset and reproduce greenlist watermark |
| 2 | Implement prompt wrapper and automated detector |
| 3 | Conduct robustness and ablation experiments |
| 4 | Perform analysis, write-up, and possibly implement a third approach |

---

## 5. Deliverables
- Codebase
- Generated corpora
- Detection scripts
- Final comparative analysis report

---

## 6. Contact

- Aime Cesaire Mugishawayo ([cmugishawayo25@amherst.edu](mailto:cmugishawayo25@amherst.edu))
- Miro Babin ([cbabin25@amherst.edu](mailto:cbabin25@amherst.edu))
- Admire Madyira ([amadyira25@amherst.edu](mailto:amadyira25@amherst.edu))

---

## 📚 References
1. [A Watermark for Large Language Models — Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226)
2. [Scalable Watermarking for Identifying Large Language Model Outputs — Lester et al. (2024)](https://arxiv.org/abs/2401.10350)
3. [Watermarking Language Models through Language Models — Zhong et al. (2024)](https://arxiv.org/abs/2411.05091)
4. [A Robust Semantics-Based Watermark for LLMs Against Paraphrasing — Ren et al. (2023)](https://arxiv.org/abs/2309.03157)
5. [Topic-Based Watermarks for Large Language Models — Wang et al. (2024)](https://arxiv.org/abs/2404.02138)

---
