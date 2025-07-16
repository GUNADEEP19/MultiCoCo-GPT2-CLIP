import os
import yaml
import time
import json
import csv
import torch
import wandb
import argparse
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

from cot import CoconutCoT
from dataset import get_cot_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, 
    CLIPModel, CLIPProcessor, BitsAndBytesConfig
)

def decode_preds(pred_ids, tokenizer):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [i for i in pred_ids if i != -100]
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    configs.lr = float(configs.lr)
    configs.weight_decay = float(configs.weight_decay)
    configs.resume = int(configs.resume)

    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name,
        config=vars(configs),
        resume=True,
        reinit=True
    )

    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = configs.save_path if hasattr(configs, "save_path") else "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Model & processor loading
    load_4bit = getattr(configs, "load_4bit", False)
    model_id = getattr(configs, "model_id", "gpt2-xl")
    clip_id = getattr(configs, "clip_id", "openai/clip-vit-base-patch32")

    print(f"Loading GPT-2 from {model_id} ...")
    gpt2_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    if load_4bit:
        print("Loading GPT-2 in 4-bit precision...")
        gpt2_kwargs["load_in_4bit"] = True
        gpt2_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    gpt2 = GPT2LMHeadModel.from_pretrained(model_id, **gpt2_kwargs)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    gpt2.resize_token_embeddings(len(tokenizer))
    print("GPT-2 loaded.")

    print(f"Loading CLIP from {clip_id} ...")
    clip = CLIPModel.from_pretrained(clip_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    print("CLIP loaded.")

    model = CoconutCoT(
        gpt2=gpt2,
        clip=clip,
        eos_token_id=tokenizer.eos_token_id
    )
    model = model.to(device)
    print("Model and processor loaded successfully!")

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, label_pad_token_id=-100, clip_model=clip, device=device)

    train_ds = get_cot_dataset(train_data, configs, tokenizer, clip_processor)
    val_ds = get_cot_dataset(val_data, configs, tokenizer, clip_processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=configs.batch_size_training,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=configs.batch_size_eval,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True
    )

    best_val_loss = float('inf')
    for epoch in range(configs.num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
                img_embeds=batch["img_embeds"].to(device) if batch["img_embeds"] is not None else None,
                position_ids=batch["position_ids"].to(device)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                    img_embeds=batch["img_embeds"].to(device) if batch["img_embeds"] is not None else None,
                    position_ids=batch["position_ids"].to(device)
                )
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} avg val loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(save_dir, "best_cot.pt"))
            print(f"Saved new best model at epoch {epoch+1}")

if __name__ == "__main__":
    main() 