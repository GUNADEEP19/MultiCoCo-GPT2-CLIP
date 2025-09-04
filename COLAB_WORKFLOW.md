# Colab Pro+ A100 Training Workflow for CoCoNuT

## Overview
This guide walks you through training CoT (Chain-of-Thought) and CoCoNuT (Continuous Thought) models on Google Colab Pro+ with A100 GPU.

## Step 1: CoT Training (Baseline)

### 1.1 Upload Code to Colab
- Upload your entire project folder to Colab
- Mount Google Drive: `from google.colab import drive; drive.mount('/content/drive')`

### 1.2 Train CoT Model
```bash
# Install requirements
pip install -r requirements.txt

# Train CoT baseline (5 epochs)
python run_cot.py args/cot.yaml
```

### 1.3 Download CoT Checkpoint
- Checkpoint saved to: `./colab_checkpoints/cot/best_cot.pt`
- Download this file from Colab disk storage
- Upload to your Google Drive for safekeeping

## Step 2: CoCoNuT Training (Continuous Thoughts)

### 2.1 Update Configuration
Edit `args/coconut.yaml`:
```yaml
load_model_path: /content/drive/MyDrive/your_folder/best_cot.pt
```

### 2.2 Train CoCoNuT Model
```bash
# Train CoCoNuT with curriculum (9 epochs total)
python run_coconut.py args/coconut.yaml
```

**Curriculum Stages:**
- **Epochs 1-3**: Stage 1 (1 latent token)
- **Epochs 4-6**: Stage 2 (2 latent tokens)  
- **Epochs 7-9**: Stage 3 (3 latent tokens)

### 2.3 Download CoCoNuT Checkpoint
- Checkpoint saved to: `./colab_checkpoints/coconut/best_coconut.pt`
- Download this file from Colab disk storage
- This is your final model for testing

## Step 3: Test Your Model

### 3.1 Evaluate on Test Data
```bash
python test_coconut.py args/coconut.yaml --checkpoint ./colab_checkpoints/coconut/best_coconut.pt
```

### 3.2 Results
- Test results saved to: `test_results.json`
- Evaluates performance across all stages (0-3 latent tokens)

## File Structure After Training
```
colab_checkpoints/
├── cot/
│   └── best_cot.pt          # Download this after CoT training
└── coconut/
    └── best_coconut.pt      # Download this after CoCoNuT training

checkpoints/                  # Main checkpoint directory
├── cot_aokvqa/
│   └── best_cot.pt
└── coconut_aokvqa/
    └── best_coconut.pt
```

## Key Features
- **Automatic Checkpoint Loading**: CoCoNuT automatically loads your CoT weights
- **Curriculum Learning**: Progressive stages from 1 to 3 latent tokens
- **Continuous Thought**: Hidden state feedback loop during training
- **Colab Optimized**: Uses gpt2-large for A100 compatibility
- **Dual Save**: Checkpoints saved to both main directory and Colab disk

## Troubleshooting
- **Memory Issues**: Use `load_4bit: true` in YAML for gpt2-xl
- **Checkpoint Not Found**: Ensure `load_model_path` points to correct CoT checkpoint
- **Slow Training**: Reduce batch size or use smaller model

## Expected Timeline
- **CoT Training**: ~2-3 hours (5 epochs)
- **CoCoNuT Training**: ~4-6 hours (9 epochs total)
- **Total**: ~6-9 hours for complete pipeline

Your model will now have learned to reason with continuous latent thoughts instead of explicit text steps!
