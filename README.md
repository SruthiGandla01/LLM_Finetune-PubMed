
# 🧠 Fine-Tuning FLAN-T5 on Biomedical Summarization (PubMed Dataset)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Hugging Face](https://img.shields.io/badge/Transformers-🤗-yellow)
![Google Colab](https://img.shields.io/badge/Google%20Colab-A100%20GPU-orange?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📘 Project Overview
This project demonstrates how to fine-tune a **Large Language Model (LLM)** — specifically **Google’s FLAN-T5-Small** — for **domain-specific text summarization** using the **PubMed Summarization** dataset.

The goal is to adapt a general instruction-tuned model to the **biomedical domain**, improving factual accuracy and relevance in generated summaries.

---

## 🧩 Key Features
- ✅ Fine-tuning FLAN-T5 on real-world biomedical abstracts (PubMed dataset)
- ✅ Data preprocessing, tokenization, and formatting for seq2seq tasks
- ✅ Hyperparameter optimization using **Ray Tune**
- ✅ Evaluation with **ROUGE metrics** and baseline comparison
- ✅ Error analysis and inference pipeline for live summarization
- ✅ Visualization of performance improvements
- ✅ Model saving, logging, and reproducibility (results, logs, wandb)

---

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Environment | Google Colab (A100 GPU) |
| Framework | 🤗 Transformers, Datasets, Evaluate |
| Optimization | Ray Tune, EarlyStoppingCallback |
| Logging | TensorBoard, Weights & Biases (wandb) |
| Visualization | Matplotlib, Pandas |
| Language | Python 3.12 |

---


---

## 🧮 Dataset
**Source:** [ccdv/pubmed-summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)  
- Domain: Biomedical research papers (articles + abstracts)  
- Total records: ~133,000    
- Input length: 512 tokens  
- Target length: 128 tokens  

---

## 🧠 Model Details
**Base model:** `google/flan-t5-small`  
**Why FLAN-T5?**
- Instruction-tuned (better generalization)  
- Lightweight (77M params → fits A100 easily)  
- Strong seq2seq summarization baseline  

### Training Configuration
| Parameter | Value |
|------------|--------|
| Learning Rate | 5e-5 |
| Batch Size | 4 |
| Epochs | 3 |
| Weight Decay | 0.01 |
| Evaluation Strategy | Epoch |
| Save Strategy | Epoch |
| Early Stopping | Patience = 2 |
| Metric | ROUGE-L |

---

## 🔍 Evaluation Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|--------|----------|----------|-----------|-------------|
| **Baseline (Pre-trained FLAN-T5)** | 0.0526 | 0.0085 | 0.0425 | 0.0475 |
| **Fine-Tuned (PubMed)** | **0.1131** | **0.0387** | **0.0886** | **0.0958** |

📈 *The fine-tuned model achieved more than 2× improvement across all metrics.*

---

## 📊 Visualization
Bar chart comparing ROUGE metrics:

ROUGE-1 ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
ROUGE-2 ▓▓▓▓▓▓▓▓▓▓▓
ROUGE-L ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
ROUGE-Lsum ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓


*(Fine-tuned scores outperform baseline in every category.)*

---

## ❌ Error Analysis

| Issue | Observation | Mitigation |
|--------|--------------|-------------|
| Truncation | Long articles >512 tokens lost details | Increase `max_input_length` to 768 |
| Factual drift | Biomedical entities occasionally paraphrased | Use domain tokenizer (e.g., SciBERT) |
| Generic phrasing | Summaries sometimes vague | Continue fine-tuning with LoRA/PEFT |

---
---
## ⚡ Setup & Reproducibility

### 🧩 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
Open in Colab
```bash
https://colab.research.google.com/github/SruthiGandla101/LLM-Finetune-PubMed/blob/main/LLM_Finetune.ipynb
```
---
### 3. Resume training
If interrupted, checkpoints auto-resume:
```bash
trainer.train(resume_from_checkpoint=True)
```
---
---
##  🧩 Lessons Learned

- Domain-specific fine-tuning significantly boosts LLM performance.
- Ray Tune enables fast hyperparameter exploration even in Colab.
- FLAN-T5-Small provides a strong, efficient baseline for biomedical text generation.

---
---
## 🚧 Future Work

- Fine-tune larger variants (FLAN-T5-Base / Large)
- Evaluate factual consistency using biomedical entity match scoring
- Deploy as a web demo (Streamlit or Hugging Face Spaces)

---
