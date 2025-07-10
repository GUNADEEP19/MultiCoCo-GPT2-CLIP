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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.cuda.amp import autocast, GradScaler


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

    # Get embeddings from base model
    base_model = model.module.base_causallm if hasattr(model, "module") else model.base_causallm
    embedding_layer = base_model.get_input_embeddings().to(device)
    token_embeddings = embedding_layer(input_ids.to(device))  # (B, L, H)

    latent_mask = (input_ids == latent_token_id)  # (B, L)
    B, L, H = token_embeddings.shape

    num_latents = latent_mask.sum(dim=1)
    if torch.any(num_latents == 0):
        # If any example has 0 latent tokens, skip injection for safety
        return token_embeddings

    # Flatten
    flat_mask = latent_mask.view(B * L)
    flat_embeds = token_embeddings.view(B * L, H)

    # Trim Z to actual latent count
    actual_latents = Z[:, :num_latents[0], :]  # assuming same across batch
    flat_embeds[flat_mask] = actual_latents.reshape(-1, H)

    return flat_embeds.view(B, L, H)



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
    configs.latent_dim = int(getattr(configs, "latent_dim", 768))
    configs.n_latents = int(getattr(configs, "n_latents", 8))
    configs.latent_lr = float(getattr(configs, "latent_lr", 5e-3))
    configs.e_steps = int(getattr(configs, "e_steps", 2))

    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name,
        config=vars(configs),
        reinit=True
    )

    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "val_loss", "val_acc", "avg_tokens"])

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

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
        model = torch.nn.DataParallel(model)

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, latent_id=LATENT_ID)

    n_train = len(train_data)
    hidden_size = model.base_causallm.config.hidden_size if hasattr(model, "base_causallm") else model.config.hidden_size
    all_latents = torch.randn(
        n_train, configs.n_latents, hidden_size,
        requires_grad=True, device=device
    )
    latent_optimizer = optim.Adam([all_latents], lr=configs.latent_lr)

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = GradScaler()

    best_val = float("inf")
    train_losses, val_losses, accuracies, token_counts = [], [], [], []

    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} | Stage {stage} ===")
        epoch_start = time.time()

        train_ds = get_cot_latent_dataset(train_data, stage, configs, START_ID, LATENT_ID, END_ID)
        loader = DataLoader(train_ds, batch_size=configs.batch_size_training, shuffle=True, collate_fn=collator)

        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=False)

        for batch in pbar:
            idxs = batch["idx"].to(device)
            Z = all_latents[idxs]

            model.eval()
            for _ in range(configs.e_steps):
                latent_optimizer.zero_grad()
                inputs_embeds = inject_latents(batch, Z, model, LATENT_ID)
                labels = batch["labels"].to(device)
                with autocast(device_type='cuda'):
                    outputs = model(
                                    input_ids=batch["input_ids"].to(device),
                                    position_ids=batch["position_ids"].to(device),
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=batch["attention_mask"].to(device),
                                    labels=labels
                                    )

                    loss_z = outputs.loss
                loss_z.backward()
                latent_optimizer.step()

            for p in all_latents.parameters():
                p.requires_grad = False
            model.train()
            optimizer.zero_grad()

            inputs_embeds = inject_latents(batch, all_latents[idxs], model, LATENT_ID)
            labels = batch["labels"].to(device)
            with autocast(device_type='cuda'):
                    outputs = model(
                        input_ids=batch["input_ids"].to(device),
                        position_ids=batch["position_ids"].to(device),
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"].to(device),
                        labels=labels
                    )
                loss_m = outputs.loss
            scaler.scale(loss_m).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss_m.item()

            for p in all_latents.parameters():
                p.requires_grad = True

            pbar.set_postfix(train_loss=loss_m.item())
            wandb_run.log({
                "train/loss_step": loss_m.item(),
                "train/epoch": epoch+1
            })

        avg_train = total_loss / len(loader)
        train_losses.append(avg_train)
        print(f"📉 Avg train loss: {avg_train:.4f}")

        model.eval()
        val_ds = get_cot_latent_dataset(val_data, stage, configs, START_ID, LATENT_ID, END_ID)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collator)

        vloss, correct, tot, tokens = 0.0, 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                idxs = batch["idx"].to(device)
                Z = all_latents[idxs]
                inputs_embeds = inject_latents(batch, Z, model, LATENT_ID)
                labels = batch["labels"].to(device)
                out = model(inputs_embeds=inputs_embeds,
                            attention_mask=batch["attention_mask"].to(device),
                            labels=labels)
                loss = out.loss
                vloss += loss.item()

                preds = out.logits.argmax(-1)
                mask = (labels != -100)
                correct += ((preds == labels) & mask).sum().item()
                tot += mask.sum().item()
                tokens += ((preds != model.config.pad_token_id) & mask).sum().item()

        avg_vl = vloss / len(val_loader)
        acc = 100 * correct / tot
        avg_tk = tokens / len(val_loader)

        val_losses.append(avg_vl)
        accuracies.append(acc)
        token_counts.append(avg_tk)

        print(f"✅ Val loss {avg_vl:.4f} | Acc {acc:.2f}% | AvgTokens {avg_tk:.1f}")

        ckpt = os.path.join(save_dir, f"checkpoint_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)
        wandb.save(ckpt)

        if avg_vl < best_val:
            best_val = avg_vl
            best_ckpt = os.path.join(save_dir, "best.pt")
            torch.save(model.state_dict(), best_ckpt)
            with open(os.path.join(save_dir, "best_info.json"), "w") as f:
                json.dump({
                    "epoch": epoch+1,
                    "val_loss": avg_vl,
                    "val_acc": acc,
                    "avg_tokens": avg_tk
                }, f, indent=2)

        wandb_run.log({
            "train/avg_loss": avg_train,
            "val/loss": avg_vl,
            "val/acc": acc,
            "val/avg_tokens": avg_tk,
            "epoch": epoch+1
        })

        fig = plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()
        fpath = os.path.join(save_dir, f"loss_{epoch+1}.png")
        fig.savefig(fpath)
        wandb.log({f"loss_plot_{epoch+1}": wandb.Image(fpath)})
        plt.close(fig)

        print(f"⏱ Epoch time: {(time.time() - epoch_start) / 60:.2f} mins")

    # Final plots
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.title("Final Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    final_loss_path = os.path.join(save_dir, "final_loss_curve.png")
    plt.savefig(final_loss_path)
    plt.close()
    wandb.log({"final/loss_curve": wandb.Image(final_loss_path)})

    plt.figure()
    plt.plot(epochs, accuracies, label="val accuracy")
    plt.title("Final Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    final_acc_path = os.path.join(save_dir, "final_accuracy_curve.png")
    plt.savefig(final_acc_path)
    plt.close()
    wandb.log({"final/accuracy_curve": wandb.Image(final_acc_path)})

    plt.figure()
    plt.plot(epochs, token_counts, label="avg tokens")
    plt.title("Final Avg Token Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Tokens")
    plt.legend()
    final_token_path = os.path.join(save_dir, "final_token_curve.png")
    plt.savefig(final_token_path)
    plt.close()
    wandb.log({"final/token_curve": wandb.Image(final_token_path)})

    wandb_run.finish()

if __name__ == "__main__":
    main()
