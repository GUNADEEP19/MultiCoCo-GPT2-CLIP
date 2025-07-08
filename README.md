# 🥥 Multimodal Latent Reasoning with GPT-2 + CLIP (Coconut-style)

This project implements a Coconut-style latent reasoning framework combining **GPT-2** (for text) and **CLIP** (for vision) to solve multimodal question answering tasks like **A-OKVQA**. It supports training with **latent tokens**, curriculum learning, and stage-wise reasoning injection.

---

## 🚀 Overview

Inspired by Meta's **Coconut** paper, this project injects latent "thought" tokens between question and answer to simulate intermediate reasoning steps. It integrates:
- **GPT-2** as the language model,
- **CLIP** as the vision encoder,
- Custom logic for staged reasoning over multiple epochs.

---

## 🧠 What Are We Training?

We are training a **multimodal latent reasoning model** to:
- Take **textual questions** and corresponding **images**,
- Insert stage-wise **latent reasoning steps** (via `<|latent|>` tokens),
- Predict the **correct answer** based on reasoning and visual context.

---

## 📁 Project Structure

| File             | Purpose                                                                 |
|------------------|-------------------------------------------------------------------------|
| `run.py`         | Entry point to train the model. Handles config parsing, checkpointing, W&B logging, curriculum training, and mixed precision AMP. |
| `coconut.py`     | Implements the custom `Coconut` model class. Wraps GPT-2 + CLIP and handles stage-wise latent token reasoning and image embedding injection. |
| `dataset.py`     | Dataset and dataloader logic. Loads `.json` files, encodes latent tokens, and handles stage-wise reasoning logic with padding and attention. |
| `args/aokvqa.yaml` | YAML configuration for training on A-OKVQA with curriculum learning and W&B tracking. |

---

## 📦 Dependencies

Install the required libraries:

```bash
pip install torch torchvision transformers datasets wandb
```

Or use Kaggle / Colab with GPU runtime.

---

## 📚 Dataset Format (A-OKVQA)

Each example in `aokvqa_train.json` should look like:

```json
{
  "image": "path/to/image.jpg",
  "question": "What is the man doing? The choices are 0 : sleeping, 1 : running, 2 : eating",
  "steps": ["The man is lying on a bed", "He looks relaxed"],
  "answer": "0"
}
```

You need both:

- `aokvqa_train.json`
- `aokvqa_validation.json`

---

## 🏁 Training Instructions

To train the model, run:

```bash
python run.py args/aokvqa.yaml
```

---

## ⚙️ Configuration (`args/aokvqa.yaml`)

```yaml
name: coconut_aokvqa_gunadeep
model_id: gpt2
project: coconut_aokvqa_guna
name: run_1_aokvqa_baseline
resume: 0
train_path: /path/to/aokvqa_train.json
val_path: /path/to/aokvqa_validation.json
coconut: true
num_epochs: 25
batch_size_training: 3
c_thought: 2
max_latent_stage: 8
epochs_per_stage: 2
lr: 5e-5
save_path: checkpoints
```

✅ To resume from checkpoint_10.pt:
```yaml
resume: 10
```

---

## 🧊 Checkpoints & Resuming

Model checkpoints are saved at:

```
checkpoints/coconut_aokvqa_gunadeep/checkpoint_{epoch}.pt
```

Resume training from a checkpoint by setting `resume:` in your YAML.

---

## 🧪 Evaluation & Inference

Evaluation loop is not included yet.

For inference, you can use the `generate()` method in `coconut.py`.

---

## 📈 Logging with Weights & Biases

Make sure you're logged in to W&B:

```bash
wandb login
```

View your runs at:  
[https://wandb.ai/gunadeep2005-pes-university/coconut_aokvqa_guna](https://wandb.ai/gunadeep2005-pes-university/coconut_aokvqa_guna)

---

## 💡 Future Enhancements

- Add evaluation metrics
- Add support for other vision models (e.g., DINOv2, BLIP)
- HuggingFace `Trainer` integration
- Inference notebook for demo

---

## 🙏 Acknowledgements

- [Coconut Paper (Meta, 2024)](https://arxiv.org/abs/2412.06769)
- [A-OKVQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Transformers](https://huggingface.co/transformers)

---

## 📝 License

MIT License.  
© 2025 Gunadeep Chennupati – PES University
