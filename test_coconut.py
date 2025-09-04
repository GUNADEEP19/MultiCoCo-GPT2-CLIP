import os
import yaml
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    CLIPModel, CLIPProcessor
)

def evaluate_model(model, test_loader, device, tokenizer, max_stage=3):
    """Evaluate model on test data with different latent stages"""
    model.eval()
    results = {}
    
    for stage in range(max_stage + 1):
        print(f"\n=== Evaluating Stage {stage} ===")
        stage_correct = 0
        stage_total = 0
        
        # Create stage-specific dataset
        test_data = get_dataset("data/Datasets/A-OKVQA/aokvqa_test.json")  # Adjust path as needed
        test_ds = get_cot_latent_dataset(
            test_data, stage, 
            type('Config', (), {'n_latents': stage, 'c_thought': 1, 'max_length': 256})(),
            tokenizer.convert_tokens_to_ids("<|start-latent|>"),
            tokenizer.convert_tokens_to_ids("<|latent|>"),
            tokenizer.convert_tokens_to_ids("<|end-latent|>"),
            tokenizer, None
        )
        
        test_loader_stage = DataLoader(
            test_ds,
            batch_size=1,  # Test one at a time for generation
            shuffle=False,
            collate_fn=MyCollator(tokenizer, label_pad_token_id=-100, clip_model=None, device=device)
        )
        
        for batch in tqdm(test_loader_stage, desc=f"Stage {stage}"):
            with torch.no_grad():
                # Generate answer
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                img_embeds = batch["img_embeds"].to(device) if batch["img_embeds"] is not None else None
                
                # Generate with continuous thought
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    img_embeds=img_embeds,
                    max_new_tokens=32
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract answer (assuming format: ... [Answer])
                # This is a simple heuristic - you might want to improve this
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text.split()[-1]  # Last token as answer
                
                # Compare with ground truth (you'll need to implement this based on your data format)
                # For now, just count total examples
                stage_total += 1
                
                # You can add more sophisticated answer comparison here
                # stage_correct += 1 if answer == ground_truth else 0
        
        accuracy = stage_correct / stage_total if stage_total > 0 else 0
        results[f"stage_{stage}"] = {
            "accuracy": accuracy,
            "correct": stage_correct,
            "total": stage_total
        }
        print(f"Stage {stage} Accuracy: {accuracy:.4f} ({stage_correct}/{stage_total})")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to CoCoNuT checkpoint")
    parser.add_argument("--test_data", default="data/Datasets/A-OKVQA/aokvqa_test.json", help="Path to test data")
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)

    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_id = getattr(configs, "model_id", "gpt2-large")
    clip_id = getattr(configs, "clip_id", "openai/clip-vit-base-patch32")
    
    print(f"Loading GPT-2 from {model_id} ...")
    gpt2 = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    
    # Add special tokens
    special_tokens_dict = {
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    gpt2.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading CLIP from {clip_id} ...")
    clip = CLIPModel.from_pretrained(clip_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)

    # Create CoCoNuT model
    model = Coconut(
        gpt2=gpt2,
        clip=clip,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    print("Model loaded successfully!")

    # Load test data
    test_data = get_dataset(args.test_data)
    print(f"Loaded {len(test_data)} test examples")

    # Evaluate model
    print("\nStarting evaluation...")
    results = evaluate_model(model, None, device, tokenizer, max_stage=3)
    
    # Save results
    results_file = "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for stage, metrics in results.items():
        print(f"{stage}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")

if __name__ == "__main__":
    main()
