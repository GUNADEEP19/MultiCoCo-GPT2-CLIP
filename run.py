import os
import yaml
import time
import json
import csv
import torch
import wandb
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
# Change tqdm import for Colab-friendly progress bars
from tqdm.notebook import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.amp import autocast, GradScaler  # switch to torch.amp

def decode_preds(pred_ids, tokenizer):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [i for i in pred_ids if i != -100]
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

def inject_latents(batch, Z, model, latent_token_id):
    input_ids = batch["input_ids"]
    B, L = input_ids.shape
    device = input_ids.device

    base = model.module.base_causallm if hasattr(model, "module") else model.base_causallm
    embedding_layer = base.get_input_embeddings().to(device)
    token_embeddings = embedding_layer(input_ids)

    latent_mask = (input_ids == latent_token_id)
    B, L, H = token_embeddings.shape
    num_latents = latent_mask.sum(dim=1)
    if torch.any(num_latents == 0):
        return token_embeddings

    # Robust per-sample injection
    for b in range(B):
        n_latent = num_latents[b].item()
        n_to_inject = min(n_latent, Z.shape[1])
        if n_to_inject > 0:
            latent_positions = torch.where(latent_mask[b])[0][:n_to_inject]
            token_embeddings[b, latent_positions, :] = Z[b, :n_to_inject, :]
    return token_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    # cast numeric configs
    configs.lr = float(configs.lr)
    configs.weight_decay = float(configs.weight_decay)
    configs.resume = int(configs.resume)
    configs.latent_dim = int(getattr(configs, "latent_dim", 768))
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

    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)

    local_ckpt_dir = "/content/checkpoints"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "train_loss", "val_loss", "val_acc", "avg_tokens"])

    # model & tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    tokenizer.add_tokens(special_tokens)
    hf_model.resize_token_embeddings(len(tokenizer))

    LATENT_ID = tokenizer.convert_tokens_to_ids("<|latent|>")
    START_ID = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    END_ID = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # Wrap in Coconut if needed
    model = hf_model
    if configs.coconut:
        model = Coconut(
            base_causallm=model,
            latent_token_id=LATENT_ID,
            start_latent_id=START_ID,
            end_latent_id=END_ID,
            eos_token_id=tokenizer.eos_token_id
        )

    # Resize embeddings on base model
    model.base_causallm.resize_token_embeddings(len(tokenizer))

    # Move entire model wrapper to device
    model = model.to(device)
    print(f"[DEBUG] Model embedding on device: {next(model.embedding.parameters()).device}")

    # Remove DataParallel (not needed for Colab single A100 GPU)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Encourage user to increase batch size for A100
    print("[INFO] For A100 GPU, consider increasing batch_size_training in your YAML config for best performance.")

    # Prepare data and model dimensions before checkpoint logic
    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, latent_id=LATENT_ID)

    n_train = len(train_data)
    base_model = model.module.base_causallm if hasattr(model, "module") else model.base_causallm
    hidden_size = base_model.config.hidden_size

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        if "ema_model" in ckpt:
            ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        all_latents = ckpt["latents"].to(device).detach().requires_grad_(True)
        start_epoch = ckpt["epoch"]
    else:
        # Initialize all_latents only if not resuming
        all_latents = torch.randn(n_train, configs.n_latents, hidden_size, requires_grad=True, device=device)

    latent_optimizer = optim.Adam([all_latents], lr=configs.latent_lr)

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = GradScaler()

    # Early stopping setup
    patience = 8
    best_val = float("inf")
    patience_counter = 0
    
    # Model EMA setup
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    train_losses, val_losses, accuracies, token_counts = [], [], [], []

    printed_checkpoint_reminder = False
    prev_best_ckpt = None  # Track previous best checkpoint (Colab only)
    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} | Stage {stage} ===")
        epoch_start = time.time()

        train_ds = get_cot_latent_dataset(train_data, stage, configs, START_ID, LATENT_ID, END_ID)
        loader = DataLoader(
            train_ds,
            batch_size=configs.batch_size_training,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True
        )

        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=True, bar_format="{l_bar}{n_fmt}/{total_fmt} [{percentage:3.0f}%]")

        for batch in pbar:
            # Move all tensors in batch to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            idxs = batch["idx"]
            Z = all_latents[idxs]

            model.eval()
            for _ in range(configs.e_steps):
                latent_optimizer.zero_grad()
                inputs_embeds = inject_latents(batch, Z, model, LATENT_ID)
                labels = batch["labels"]
                with autocast(device_type='cuda'):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        position_ids=batch.get("position_ids"),
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"],
                        labels=labels
                    )
                    loss_z = outputs.loss
                loss_z.backward()
                latent_optimizer.step()

            # Disable gradients for all_latents
            all_latents.requires_grad = False
            model.train()
            optimizer.zero_grad()

            inputs_embeds = inject_latents(batch, all_latents[idxs], model, LATENT_ID)
            labels = batch["labels"]
            with autocast(device_type='cuda'):
                outputs = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch["attention_mask"],
                    labels=labels
                )
                loss_m = outputs.loss
            scaler.scale(loss_m).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update EMA model
            with torch.no_grad():
                for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                    ema_param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data
            
            total_loss += loss_m.item()

            all_latents.requires_grad = True
            pbar.set_postfix(train_loss=loss_m.item())
            wandb_run.log({"train/loss_step": loss_m.item(), "train/epoch": epoch+1})

        avg_train = total_loss / len(loader)
        train_losses.append(avg_train)
        print(f"📉 Avg train loss: {avg_train:.4f}")

        model.eval()
        val_ds = get_cot_latent_dataset(val_data, stage, configs, START_ID, LATENT_ID, END_ID)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True
        )

        vloss, correct, tot, tokens = 0.0, 0, 0, 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc="Validating", leave=True, bar_format="{l_bar}{n_fmt}/{total_fmt} [{percentage:3.0f}%]")
            for batch in vbar:
                # Move all tensors in batch to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                idxs = batch["idx"]
                Z = all_latents[idxs]
                inputs_embeds = inject_latents(batch, Z, model, LATENT_ID)
                labels = batch["labels"]
                if batch.get("position_ids") is not None:
                    batch["position_ids"] = batch["position_ids"].to(device)
                out = model(
                    input_ids=batch["input_ids"],
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch["attention_mask"],
                    position_ids=batch.get("position_ids"),
                    labels=labels
                )
                loss = out.loss
                vloss += loss.item()
                preds = out.logits.argmax(-1)
                mask = (labels != -100)
                correct += ((preds == labels) & mask).sum().item()
                tot += mask.sum().item()
                tokens += ((preds != base_model.config.pad_token_id) & mask).sum().item()
                vbar.set_postfix(val_loss=loss.item())

        avg_vl = vloss / len(val_loader)
        acc = 100 * correct / tot
        avg_tk = tokens / len(val_loader)

        val_losses.append(avg_vl)
        accuracies.append(acc)
        token_counts.append(avg_tk)

        print(f"✅ Val loss {avg_vl:.4f} | Acc {acc:.2f}% | AvgTokens {avg_tk:.1f}")

        # Append metrics to CSV after each epoch
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, stage, avg_train, avg_vl, acc, avg_tk])

        # After validation, save only best.pt to Colab local storage, not Drive
        if avg_vl < best_val:
            # Delete previous best checkpoint in Colab if it exists
            if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
            best_val = avg_vl
            patience_counter = 0
            # Save best EMA model to Colab local storage with informative filename
            best_ckpt_name = f"best_epoch{epoch+1}_valloss{avg_vl:.4f}.pt"
            best_ckpt = os.path.join(local_ckpt_dir, best_ckpt_name)
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "latents": all_latents,
                "epoch": epoch+1
            }, best_ckpt)
            # Do NOT save best.pt to Drive or W&B
            # Save best_info.json in both Drive and Colab local storage
            best_info = {"epoch": epoch+1, "val_loss": avg_vl, "val_acc": acc, "avg_tokens": avg_tk}
            with open(os.path.join(save_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)
            with open(os.path.join(local_ckpt_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)
            prev_best_ckpt = best_ckpt  # Update tracker
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered! No improvement for {patience} epochs.")
                break

        # Save regular checkpoint to local Colab storage and W&B only
        ckpt_name = f"checkpoint_epoch{epoch+1}_valloss{avg_vl:.4f}.pt"
        ckpt = os.path.join(local_ckpt_dir, ckpt_name)
        torch.save({
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "latents": all_latents,
            "epoch": epoch+1
        }, ckpt)
        wandb.save(ckpt, base_path=local_ckpt_dir)

        # After saving the regular checkpoint, print a reminder for the user only once
        if not printed_checkpoint_reminder:
            print(f"[INFO] To keep a specific checkpoint (e.g., {ckpt_name}), copy it to Drive before your Colab session ends:")
            print(f"!cp /content/checkpoints/{ckpt_name} {save_dir}/")
            printed_checkpoint_reminder = True

        fig = plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()
        # Remove saving to Drive
        # fpath = os.path.join(save_dir, f"loss_{epoch+1}.png")
        # fig.savefig(fpath)
        wandb.log({f"loss_plot_{epoch+1}": wandb.Image(fig)})
        plt.close(fig)
        print(f"⏱ Epoch time: {(time.time() - epoch_start)/60:.2f} mins")

    wandb_run.finish()

if __name__ == "__main__":
    main()
