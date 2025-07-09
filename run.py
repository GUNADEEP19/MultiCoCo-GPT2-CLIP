import os
import yaml
import time
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
from torch.amp import autocast, GradScaler  # ← keep this, no device_type arg


def decode_preds(pred_ids, tokenizer):
    return tokenizer.decode(pred_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    # Load config
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    configs.lr = float(configs.lr)
    configs.weight_decay = float(configs.weight_decay)
    configs.resume = int(configs.resume)

    # W&B
    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name,
        config=vars(configs),
        reinit=True
    )

    # Seed and device
    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint dir
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Special tokens
    special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    LATENT_ID = tokenizer.convert_tokens_to_ids("<|latent|>")
    START_ID = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    END_ID = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if configs.coconut:
        model = Coconut(
            base_causallm=model,
            latent_token_id=LATENT_ID,
            start_latent_id=START_ID,
            end_latent_id=END_ID,
            eos_token_id=tokenizer.eos_token_id
        )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"✅ Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Load datasets
    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, latent_id=LATENT_ID)

    # Optimizer + AMP (💡 No device_type param here)
    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = GradScaler()  # ← old compatible style

    global_step = 0
    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        print(f"\n🎯 Epoch {epoch + 1}/{configs.num_epochs} | Curriculum Stage: {stage}")
        epoch_start = time.time()

        train_ds = get_cot_latent_dataset(train_data, stage, configs, START_ID, LATENT_ID, END_ID)
        loader = DataLoader(train_ds, batch_size=configs.batch_size_training, shuffle=True, collate_fn=collator)

        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}

            with autocast():
                outputs = model(**batch)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                if loss.dim() != 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix(loss=round(loss.item(), 4))

            wandb_run.log({
                "train/loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/step": global_step,
            })

        avg_train_loss = epoch_loss / len(loader)
        print(f"📉 Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_ds = get_cot_latent_dataset(val_data, stage, configs, START_ID, LATENT_ID, END_ID)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator)

        val_loss_total = 0
        correct = 0
        total = 0

        print("🔍 Sample Predictions:")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
                batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
                outputs = model(**batch)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                if loss.dim() != 0:
                    loss = loss.mean()
                val_loss_total += loss.item()

                preds = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]
                correct += (preds == labels).sum().item()
                total += labels.numel()

                if i < 3:
                    try:
                        print(f"\nSample {i+1}")
                        print("Q:", decode_preds(batch["input_ids"], tokenizer))
                        print("Pred:", decode_preds(preds, tokenizer))
                        print("Answer:", decode_preds(labels, tokenizer))
                    except Exception as e:
                        print(f"(⚠️ Decode error: {e})")

        val_loss_avg = val_loss_total / len(val_loader)
        val_accuracy = 100.0 * correct / total
        print(f"\n✅ Val Loss: {val_loss_avg:.4f} | Accuracy: {val_accuracy:.2f}%")

        # Log & save
        wandb_run.log({
            "train/avg_loss": avg_train_loss,
            "val/loss": val_loss_avg,
            "val/accuracy": val_accuracy,
            "val/epoch": epoch + 1,
            "curriculum/stage": stage
        })

        ckpt_epoch = epoch + 1
        ckpt_path = os.path.join(save_dir, f"checkpoint_{ckpt_epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)

        elapsed = time.time() - epoch_start
        print(f"⏱️ Time Taken: {elapsed/60:.2f} mins")

    wandb_run.finish()


if __name__ == "__main__":
    main()
