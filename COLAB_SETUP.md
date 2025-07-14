# 🚀 COCONUT LLaVA-1.5-7B Colab Setup Guide

This guide helps you set up and run COCONUT training on LLaVA-1.5-7B in Google Colab with A-OKVQA dataset.

## 📚 Reference Resources

- **[LLaVA-1.5-7B Model](https://huggingface.co/llava-hf/llava-1.5-7b-hf)**: Official model repository
- **[Colab Demo](https://colab.research.google.com/#scrollTo=tXneIeME9zFm&fileId=https%3A//huggingface.co/llava-hf/llava-1.5-7b-hf.ipynb)**: Free-tier LLaVA demo
- **[Training Notebook](https://colab.research.google.com/drive/1qsl6cd2c8gGtEW1xV5io7S8NHh-Cp1TV?usp=sharing)**: Your training environment

## 🛠️ Step 1: Environment Setup

### Install Dependencies
```python
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers>=4.35.3
!pip install accelerate
!pip install wandb
!pip install tqdm
!pip install matplotlib
!pip install scikit-learn

# Optional optimizations (uncomment if needed)
# !pip install bitsandbytes  # For 4-bit quantization
# !pip install flash-attn     # For Flash Attention 2

# Restart runtime after installation
print("⚠️  Please restart the Colab runtime after installation!")
```

### HuggingFace Authentication
```python
from huggingface_hub import login
login(token="your_hf_token_here")  # Create at https://huggingface.co/settings/tokens
```

### Weights & Biases Setup
```python
import wandb
wandb.login()
```

## 📁 Step 2: Dataset Preparation

### Upload A-OKVQA Dataset
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create dataset directory
!mkdir -p /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/
!mkdir -p /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/images/

# Upload your dataset files to:
# /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/aokvqa_train.json
# /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/aokvqa_validation.json
# /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/images/ (all images)
```

## ⚙️ Step 3: Configuration

### Basic Configuration (`args/aokvqa.yaml`)
```yaml
name: run_1_aokvqa_baseline
model_id: llava-hf/llava-1.5-7b-hf
seed: 42
bf16: true
project: Coconut_Aokvqa_Guna_LLaVA-1.5-7B

# Save checkpoints to Google Drive (persistent)
save_path: /content/drive/MyDrive/COCONUT/checkpoints/coconut_aokvqa_gunadeep

# Dataset paths
train_path: /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/aokvqa_train.json
val_path: /content/MultiModal-COCONUT-GPT2-CLIP-/data/Datasets/A-OKVQA/aokvqa_validation.json

# Curriculum configs
c_thought: 1
uniform_prob: 0.1

# Latent-space configs
latent_dim: 4096  # Matches LLaVA-1.5-7B hidden size
latent_lr: 1e-3
e_steps: 3

# Training configs for A100 GPU
num_epochs: 28
batch_size_training: 48
lr: 1e-4
weight_decay: 0.0

max_length: 512
max_latent_stage: 7
epochs_per_stage: 4
```

### Advanced Configuration (with optimizations)
```yaml
# Add these lines to enable optimizations:
load_4bit: true              # 4-bit quantization (saves memory)
use_flash_attention_2: true  # Flash Attention 2 (faster training)
```

## 🏃‍♂️ Step 4: Training

### Upload Code to Colab
```python
# Upload your COCONUT code files to Colab
# - coconut.py
# - dataset.py
# - run.py
# - utils.py
# - args/aokvqa.yaml
```

### Test Model Loading (Recommended)
```python
# First, test if model loading works correctly
!python test_model_loading.py
```

### Start Training
```python
# If tests pass, start training
!python run.py args/aokvqa.yaml
```

## 📊 Step 5: Monitoring

### Weights & Biases Dashboard
- **Project**: `Coconut_Aokvqa_Guna_LLaVA-1.5-7B`
- **Metrics**: Training loss, validation accuracy, latent trajectories
- **Visualizations**: t-SNE plots, token histograms, sample predictions

### Local Monitoring
```python
# Check training progress
!tail -f /content/checkpoints/metrics.csv

# Monitor GPU usage
!nvidia-smi
```

## 💾 Step 6: Checkpoint Management

### Save Important Checkpoints
```python
# Copy best checkpoint to Drive
!cp /content/checkpoints/best_*.pt /content/drive/MyDrive/COCONUT/checkpoints/coconut_aokvqa_gunadeep/

# Save tokenizer with special tokens
!cp -r /content/checkpoints/tokenizer /content/drive/MyDrive/COCONUT/checkpoints/coconut_aokvqa_gunadeep/
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `batch_size_training` to 24 or 16
   - Enable `load_4bit: true`
   - Reduce `max_length` to 256

2. **HuggingFace Authentication Error**
   - Ensure your token has read access to gated models
   - Create token at: https://huggingface.co/settings/tokens

3. **Slow Training**
   - Enable `use_flash_attention_2: true`
   - Increase `batch_size_training` if memory allows
   - Use A100 GPU runtime

4. **Dataset Loading Issues**
   - Check image paths in JSON files
   - Ensure all images are in the correct directory
   - Verify JSON format matches expected structure

### Performance Tips

- **A100 GPU**: Use batch size 48-64
- **V100 GPU**: Use batch size 16-24
- **T4 GPU**: Use batch size 4-8 with 4-bit quantization

## 📈 Expected Results

With proper setup, you should see:
- **Training Loss**: Decreasing from ~4.0 to ~1.5
- **Validation Accuracy**: Improving from ~20% to ~60%+
- **Latent Trajectories**: Meaningful evolution across curriculum stages
- **Memory Usage**: ~15-20GB on A100 GPU

## 🎯 Next Steps

1. **Experiment with curriculum stages**: Adjust `max_latent_stage` and `epochs_per_stage`
2. **Try different latent dimensions**: Test `latent_dim` values
3. **Optimize hyperparameters**: Tune learning rates and batch sizes
4. **Add evaluation metrics**: Implement additional accuracy measures
5. **Deploy for inference**: Create inference script for new questions

---

**Happy Training! 🥥🚀**