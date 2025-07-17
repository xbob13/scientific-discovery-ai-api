# 🧠 Scientific Discovery AI – Full System

This repository includes both:

- ✅ A Google Colab-compatible **training script** to fine-tune a language model on 1000+ arXiv papers
- 🌐 A **FastAPI inference server** deployable on Render to serve your trained model

---

## 🏗️ Structure

```
📁 scientific-discovery-ai-full/
├── api/                     # Render API backend
│   ├── app.py
│   ├── requirements.txt
│   └── render.yaml
├── scientific_discovery_ai_training.py  # Colab training pipeline
└── README.md
```

---

## 🔧 Render API Endpoint

Deploy `api/` to [Render](https://render.com/) and POST to `/generate` with a JSON body:

```json
{ "prompt": "What connects quantum mechanics and biology?" }
```

---

## 🧪 Model Training (Colab)

Use `scientific_discovery_ai_training.py` to:

- Install dependencies
- Collect papers from arXiv
- Build cross-disciplinary dataset
- Fine-tune a model with LoRA
- Save model for Hugging Face or Render

---

## 📦 Hugging Face

Publish your trained model to Hugging Face:
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./scientific-discovery-ai")
model = AutoModelForCausalLM.from_pretrained("./scientific-discovery-ai")

model.push_to_hub("your-username/scientific-discovery-ai")
tokenizer.push_to_hub("your-username/scientific-discovery-ai")
```

---

