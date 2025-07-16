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

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
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
    configs.latent_dim = int(getattr(configs, "latent_dim", 1600))
    configs.n_latents = int(getattr(configs, "n_latents", 8))
    configs.latent_lr = float(getattr(configs, "latent_lr", 5e-3))
    configs.e_steps = int(getattr(configs, "e_steps", 2))

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
    local_ckpt_dir = os.path.join("./checkpoints")
    os.makedirs(local_ckpt_dir, exist_ok=True)

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
    special_tokens_dict = {
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    gpt2.resize_token_embeddings(len(tokenizer))
    print("GPT-2 loaded.")

    print(f"Loading CLIP from {clip_id} ...")
    clip = CLIPModel.from_pretrained(clip_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    print("CLIP loaded.")

    model = Coconut(
        gpt2=gpt2,
        clip=clip,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id
    )
    model = model.to(device)
    print("Model and processor loaded successfully!")

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, label_pad_token_id=-100, clip_model=clip, device=device)

    n_train = len(train_data)
    hidden_size = gpt2.config.hidden_size

    # Checkpoint handling
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]

    patience = 8
    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    max_latent_stage = getattr(configs, "max_latent_stage", 7)
    epochs_per_stage = getattr(configs, "epochs_per_stage", 4)
    for epoch in range(start_epoch, configs.num_epochs):
        stage = min(epoch // epochs_per_stage, max_latent_stage-1)
        n_latents = stage + 1
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} | Stage {stage} | n_latents {n_latents} ===")
        epoch_start = time.time()

        configs.n_latents = n_latents
        train_ds = get_cot_latent_dataset(train_data, stage, configs,
                                          tokenizer.convert_tokens_to_ids("<|start-latent|>"),
                                          tokenizer.convert_tokens_to_ids("<|latent|>"),
                                          tokenizer.convert_tokens_to_ids("<|end-latent|>"),
                                          tokenizer,
                                          clip_processor)
        val_ds = get_cot_latent_dataset(val_data, stage, configs,
                                        tokenizer.convert_tokens_to_ids("<|start-latent|>"),
                                        tokenizer.convert_tokens_to_ids("<|latent|>"),
                                        tokenizer.convert_tokens_to_ids("<|end-latent|>"),
                                        tokenizer,
                                        clip_processor)
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
                position_ids=batch["position_ids"].to(device),
                latents=None # You may want to add latent logic here if needed
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
                    position_ids=batch["position_ids"].to(device),
                    latents=None # You may want to add latent logic here if needed
                )
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} avg val loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(save_dir, "best_coconut.pt"))
            print(f"Saved new best model at epoch {epoch+1}")

if __name__ == "__main__":
    main() 