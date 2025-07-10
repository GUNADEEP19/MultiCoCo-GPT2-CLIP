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
from torch.amp import autocast, GradScaler

def decode_preds(pred_ids, tokenizer):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [i for i in pred_ids if i != -100]
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

def inject_latents(batch, Z, model):
    """
    Replace the special <|latent|> token positions in the batch's input_ids
    with the continuous latents Z, producing inputs_embeds for the model.
    Assumes:
      - batch["input_ids"] is (B, L)
      - Z is (B, n_latents, hidden_size)
      - All latent tokens are contiguous and collated by MyCollator.
    """
    # 1) get token embeddings for all input_ids
    input_ids = batch["input_ids"]           # (B, L)
    token_embeds = model.module if hasattr(model, "module") else model
    token_embeds = token_embeds.get_input_embeddings()(input_ids.to(model.device))
    # 2) find where the latent tokens are placed (by convention MyCollator reserves them)
    #    we assume they occupy positions `latent_start:latent_end` in the sequence
    latent_mask = (input_ids == batch["latent_token_id"])  # boolean mask (B, L)
    B, L, H = token_embeds.shape

    # reshape Z to match those positions
    # we assume exactly n_latents per example
    Z = Z.to(model.device)
    assert Z.shape[1] == latent_mask.sum(dim=1)[0], \
        "Number of inferred latents does not match number of <|latent|> tokens"

    # scatter Z into token_embeds where latent_mask is True
    flat_mask = latent_mask.view(B * L)
    flat_embeds = token_embeds.view(B * L, H)
    flat_embeds[flat_mask] = Z.reshape(-1, H)
    return flat_embeds.view(B, L, H)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    # ─── Load config ─────────────────────────────────────────────────
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    configs.lr              = float(configs.lr)
    configs.weight_decay    = float(configs.weight_decay)
    configs.resume          = int(configs.resume)
    configs.latent_dim      = int(configs.latent_dim)     # hidden size of each latent vector
    configs.n_latents       = int(configs.n_latents)      # how many latents per example
    configs.latent_lr       = float(configs.latent_lr)    # E‑step learning rate
    configs.e_steps         = int(configs.e_steps)        # how many E‑steps per batch

    # ─── W&B Init ────────────────────────────────────────────────────
    wandb.login()
    wandb_run = wandb.init(
        project = configs.project,
        name    = configs.name,
        config  = vars(configs),
        reinit  = True
    )

    # ─── Set seed & device ───────────────────────────────────────────
    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Save path & resume ──────────────────────────────────────────
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)
    start_epoch = configs.resume
    ckpt_path   = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") \
                    if start_epoch>0 else None

    # Prepare metrics CSV
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch","stage","val_loss","val_acc","avg_tokens"])

    # ─── Load model & tokenizer ──────────────────────────────────────
    model     = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # add special tokens for latent injection
    special_tokens = ["<|start-latent|>","<|latent|>","<|end-latent|>"]
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
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path,map_location=device))

    # ─── Load data & collator ────────────────────────────────────────
    train_data = get_dataset(configs.train_path)
    val_data   = get_dataset(configs.val_path)
    collator   = MyCollator(tokenizer, latent_id=LATENT_ID)

    # ─── Allocate one latent per example ─────────────────────────────
    n_train = len(train_data)
    # shape (n_train, n_latents, hidden_size)
    all_latents = torch.randn(
        n_train, configs.n_latents, model.config.hidden_size,
        requires_grad=True, device=device
    )
    latent_optimizer = optim.Adam([all_latents], lr=configs.latent_lr)

    # ─── Optimizer & AMP ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=configs.lr,
                            weight_decay=configs.weight_decay)
    scaler    = GradScaler()

    best_val = float("inf")

    # ─── Training loop with EM steps ─────────────────────────────────
    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} | Stage {stage} ===")
        epoch_start = time.time()

        # prepare dataloader with indices
        train_ds = get_cot_latent_dataset(
            train_data, stage, configs, START_ID, LATENT_ID, END_ID
        )
        loader = DataLoader(
            train_ds,
            batch_size = configs.batch_size_training,
            shuffle    = True,
            collate_fn = collator
        )

        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=False)

        for batch in pbar:
            idxs = batch["idx"].to(device)  # indices of examples
            Z    = all_latents[idxs]        # (B, n_latents, hidden)

            # — E‑step: infer Z given fixed model —
            model.eval()
            for _ in range(configs.e_steps):
                latent_optimizer.zero_grad()
                inputs_embeds = inject_latents(batch, Z, model)
                labels = batch["labels"].to(device)
                with autocast():
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"].to(device),
                        labels=labels
                    )
                    loss_z = outputs.loss
                loss_z.backward()
                latent_optimizer.step()

            # — M‑step: update model given fixed Z —
            for p in all_latents.parameters():
                p.requires_grad = False
            model.train()
            optimizer.zero_grad()

            inputs_embeds = inject_latents(batch, all_latents[idxs], model)
            labels = batch["labels"].to(device)
            with autocast():
                outputs = model(
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

            # W&B logging of per‑step
            wandb_run.log({
                "train/loss_step": loss_m.item(),
                "train/epoch": epoch+1
            })

        avg_train = total_loss / len(loader)
        print(f"📉 Avg train loss: {avg_train:.4f}")

        # ─── Validation ──────────────────────────────────────────────
        model.eval()
        val_ds     = get_cot_latent_dataset(
            val_data, stage, configs, START_ID, LATENT_ID, END_ID
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, collate_fn=collator
        )

        vloss = 0.0
        correct = tot = tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                idxs = batch["idx"].to(device)
                Z    = all_latents[idxs]

                inputs_embeds = inject_latents(batch, Z, model)
                labels = batch["labels"].to(device)
                out = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch["attention_mask"].to(device),
                    labels=labels
                )
                loss = out.loss
                vloss += loss.item()

                preds = out.logits.argmax(-1)
                mask  = (labels != -100)
                correct += ((preds==labels)&mask).sum().item()
                tot     += mask.sum().item()
                tokens  += ((preds!=model.config.pad_token_id)&mask).sum().item()

        avg_vl = vloss / len(val_loader)
        acc    = 100*correct/tot
        avg_tk = tokens/len(val_loader)
        print(f"✅ Val loss {avg_vl:.4f} | Acc {acc:.2f}% | AvgTokens {avg_tk:.1f}")

        # ─── Checkpoint ─────────────────────────────────────────────
        ckpt = os.path.join(save_dir, f"checkpoint_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)
        wandb.save(ckpt)

        if avg_vl < best_val:
            best_val = avg_vl
            best_ckpt = os.path.join(save_dir, "best.pt")
            torch.save(model.state_dict(), best_ckpt)
            with open(os.path.join(save_dir,"best_info.json"), "w") as f:
                json.dump({
                    "epoch": epoch+1,
                    "val_loss": avg_vl,
                    "val_acc": acc,
                    "avg_tokens": avg_tk
                }, f, indent=2)

        # ─── Log & plot ────────────────────────────────────────────
        wandb_run.log({
            "train/avg_loss": avg_train,
            "val/loss":      avg_vl,
            "val/acc":       acc,
            "val/avg_tokens":avg_tk,
            "epoch":         epoch+1
        })

        # save per‑epoch loss plot
        fig = plt.figure()
        plt.plot(avg_train, label="train")
        plt.plot(avg_vl,    label="val")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()
        fpath = os.path.join(save_dir, f"loss_{epoch+1}.png")
        fig.savefig(fpath)
        wandb.log({f"loss_plot_{epoch+1}": wandb.Image(fpath)})
        plt.close(fig)

        print(f"⏱ Epoch time: {(time.time()-epoch_start)/60:.2f} mins")

    wandb_run.finish()
# ─── Final Summary Plots ─────────────────────────────────────────────
epochs = list(range(1, configs.num_epochs + 1))

# Final Loss Plot
plt.figure()
plt.plot(epochs, history_train_loss, label="train")
plt.plot(epochs, history_val_loss,   label="val")
plt.title("Final Loss Curve")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
final_loss_path = os.path.join(save_dir, "final_loss_curve.png")
plt.savefig(final_loss_path)
plt.close()
wandb.log({"final/loss_curve": wandb.Image(final_loss_path)})

# Final Accuracy Plot
plt.figure()
plt.plot(epochs, history_accuracy, label="val accuracy")
plt.title("Final Accuracy Curve")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
final_acc_path = os.path.join(save_dir, "final_accuracy_curve.png")
plt.savefig(final_acc_path)
plt.close()
wandb.log({"final/accuracy_curve": wandb.Image(final_acc_path)})

# Final Token Count Plot
plt.figure()
plt.plot(epochs, history_tokens, label="avg tokens")
plt.title("Final Avg Token Curve")
plt.xlabel("epoch")
plt.ylabel("tokens")
plt.legend()
final_token_path = os.path.join(save_dir, "final_token_curve.png")
plt.savefig(final_token_path)
plt.close()
wandb.log({"final/token_curve": wandb.Image(final_token_path)})


if __name__=="__main__":
    main()
