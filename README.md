# COCONUT: Curriculum-Optimized CoT Using Neural Uncertainty Tokens

COCONUT is a research codebase for training and evaluating a multimodal model that combines GPT-2 (language) and CLIP (vision) for step-by-step reasoning on datasets like A-OKVQA and ScienceQA. The model uses curriculum learning and latent token optimization to improve chain-of-thought (CoT) reasoning.

## Features
- Curriculum learning for latent token insertion
- EM-style latent optimization
- Multimodal support (text + images)
- t-SNE visualization of latent embeddings
- Checkpointing and metrics logging

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- matplotlib
- numpy
- wandb (for experiment tracking)
- PIL (Pillow)

Install dependencies:
```bash
pip install -r coconut/requirements.txt
```

## Datasets
- Place A-OKVQA and ScienceQA datasets in `coconut/data/Datasets/` as structured in the repo.
- Update dataset paths in the YAML config files under `coconut/args/` as needed.

## Training
Run training with a config file (YAML):
```bash
python coconut/run.py coconut/args/aokvqa.yaml
```

- Checkpoints and metrics will be saved to the path specified in the config file.
- You can resume training by setting the `resume` parameter in the config.

## Configuration
- All training and model parameters are controlled via YAML files in `coconut/args/`.
- Example configs: `aokvqa.yaml`, `scienceqa.yaml`

## Model Structure
- `coconut/coconut.py`: Model definition (Coconut class)
- `coconut/dataset.py`: Dataset loading and preprocessing
- `coconut/run.py`: Training and evaluation loop

## Visualization
- The code supports t-SNE plots and sample predictions for analysis.
- Plots and tables are logged using wandb (optional, can be disabled if not needed).

## Notes
- This codebase is intended for research and educational purposes.
- For best results, use a GPU with sufficient memory (gpt2-xl and CLIP are large models).

## License
See `coconut/LICENSE` for license information. 