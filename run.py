import os
import yaml
import time
import json
import csv
import torch
import wandb
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.amp import autocast, GradScaler

def decode_preds(pred_ids, tokenizer):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [id for id in pred_ids if id != -100]
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    # ─── Load config ─────────────────────────────────────────────────
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    configs.lr = float(configs.lr)
    configs.weight_decay = float(configs.weight_decay)
    configs.resume = int(configs.resume)

    # ─── W&B Init ────────────────────────────────────────────────────
    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name,
        config=vars(configs),
        reinit=True
    )

    # ─── Set seed & device ───────────────────────────────────────────
    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Save path & resume ──────────────────────────────────────────
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    # Prepare metrics CSV
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "val_loss", "val_accuracy", "avg_tokens"])

    # ─── Load model & tokenizer ──────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Special Tokens
    special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    LATENT_ID = tokenizer.convert_tokens_to_ids("<|latent|>")
    START_ID  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    END_ID    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if configs.coconut:
        model = Coconut(
            base_causallm   = model,
            latent_token_id = LATENT_ID,
            start_latent_id = START_ID,
            end_latent_id   = END_ID,
            eos_token_id    = tokenizer.eos_token_id
        )

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ─── Dataset & Collator ──────────────────────────────────────────
    train_data = get_dataset(configs.train_path)
    val_data   = get_dataset(configs.val_path)
    collator   = MyCollator(tokenizer, latent_id=LATENT_ID)

    # ─── Optimizer & AMP ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler    = GradScaler('cuda')

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        print(f"\n🎯 Epoch {epoch+1}/{configs.num_epochs} | Curriculum Stage: {stage}")
        epoch_start = time.time()

        # ─── Training ────────────────────────────────────────────────
        train_ds = get_cot_latent_dataset(train_data, stage, configs, START_ID, LATENT_ID, END_ID)
        loader   = DataLoader(train_ds, batch_size=configs.batch_size_training,
                              shuffle=True, collate_fn=collator)

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k,v in batch.items() if k!="idx"}

            with autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                loss = loss.mean() / configs.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += (loss.item() * configs.gradient_accumulation_steps)
            global_step += 1

            if (step+1) % configs.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(loss=round(loss.item()*configs.gradient_accumulation_steps,4))
            wandb_run.log({
                "train/loss":  loss.item()*configs.gradient_accumulation_steps,
                "train/epoch": epoch+1,
                "train/step":  global_step
            })

        avg_train_loss = epoch_loss / len(loader)
        print(f"📉 Avg Train Loss: {avg_train_loss:.4f}")

        # ─── Validation ──────────────────────────────────────────────
        model.eval()
        val_ds     = get_cot_latent_dataset(val_data, stage, configs, START_ID, LATENT_ID, END_ID)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator)

        val_loss_total = 0.0
        correct = 0
        total   = 0
        total_tokens_generated = 0  # NEW: accumulate total tokens

        print("🔍 Sample Predictions:")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
                batch   = {k: v.to(device) for k,v in batch.items() if k!="idx"}
                outputs = model(**batch)
                loss    = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                val_loss_total += loss.mean().item()

                preds  = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                mask   = labels != -100
                correct += ((preds==labels)&mask).sum().item()
                total   += mask.sum().item()

                # ✅ Count generated tokens (exclude padding & -100)
                token_mask = (preds != tokenizer.pad_token_id) & (labels != -100)
                total_tokens_generated += token_mask.sum().item()

                if i<3:
                    print(f"\nSample {i+1}")
                    print("Q:   ", decode_preds(batch["input_ids"], tokenizer))
                    print("Pred:", decode_preds(preds, tokenizer))
                    print("Ans: ", decode_preds(labels, tokenizer))

        val_loss_avg  = val_loss_total / len(val_loader)
        val_accuracy  = 100.0 * correct/total if total>0 else 0.0
        avg_tokens_generated = total_tokens_generated / len(val_loader)  # NEW: average per example

        print(f"\n✅ Val Loss: {val_loss_avg:.4f} | "
              f"Accuracy: {val_accuracy:.2f}% | "
              f"Avg Tokens: {avg_tokens_generated:.2f}")

        # ─── Log metrics ───────────────────────────────────────────────
        wandb_run.log({
            "train/avg_loss":       avg_train_loss,
            "val/loss":             val_loss_avg,
            "val/accuracy":         val_accuracy,
            "val/generated_tokens": avg_tokens_generated,
            "val/epoch":            epoch+1,
            "curriculum":           stage
        })

        # ─── Append to CSV ────────────────────────────────────────────
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                stage,
                f"{val_loss_avg:.4f}",
                f"{val_accuracy:.2f}",
                f"{avg_tokens_generated:.2f}"
            ])

        # ─── Checkpoint Saving ───────────────────────────────────────
        ckpt_epoch = epoch+1
        ckpt_path  = os.path.join(save_dir, f"checkpoint_{ckpt_epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)

        # ─── Best Model & Info ───────────────────────────────────────
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_path = os.path.join(save_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🔥 New best model saved: {best_path}")

            # ✅ Save JSON with epoch & metrics
            best_info = {
                "epoch": epoch+1,
                "val_loss": round(val_loss_avg, 4),
                "val_accuracy": round(val_accuracy, 2),
                "avg_tokens": round(avg_tokens_generated, 2)
            }
            best_info_path = os.path.join(save_dir, "best_info.json")
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)
            print(f"📄 Best info saved at: {best_info_path}")

        elapsed = time.time() - epoch_start
        print(f"⏱️ Time taken: {elapsed/60:.2f} mins")

    wandb_run.finish()

if __name__ == "__main__":
    main()
