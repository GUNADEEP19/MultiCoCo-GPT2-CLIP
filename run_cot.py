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
from tqdm.notebook import tqdm
from transformers import LlavaForCausalLM, LlavaProcessor
import copy
from PIL import Image

from utils import Config, set_seed
from torch.cuda.amp import autocast, GradScaler

def get_dataset(path, max_size=1_000_000_00):
    with open(path, "r") as f:
        raw = json.load(f)[:max_size]
    return [
        {
            "question": d["question"],
            "steps": d["steps"],
            "answer": d["answer"],
            "image": d.get("image", None),
            "idx": i
        }
        for i, d in enumerate(raw)
    ]

def decode_preds(pred_ids, processor):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [i for i in pred_ids if i != -100]
    return processor.tokenizer.decode(pred_ids, skip_special_tokens=True)

def get_cot_dataset(base_dataset, processor, max_len=512):
    class CoTDataset(torch.utils.data.Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            # Compose full CoT prompt: question + all steps + answer
            text = ex["question"]
            if len(ex["steps"]) > 0:
                text += " " + " ".join(ex["steps"])
            text += " " + ex["answer"]
            image_path = ex.get("image", None)
            if image_path is not None:
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = processor.image_processor(img, return_tensors="pt")["pixel_values"][0]
                except Exception as e:
                    print(f"[WARNING] Could not load image {image_path}: {e}")
                    img_tensor = None
            else:
                img_tensor = None
            processed = processor(text=text, images=img_tensor, return_tensors="pt", padding="max_length", max_length=max_len)
            input_ids = processed.input_ids[0]
            attention_mask = processed.attention_mask[0]
            pixel_values = processed.pixel_values[0] if img_tensor is not None else None
            position_ids = torch.arange(len(input_ids)).clamp(max=max_len-1)
            labels = input_ids.clone()
            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "pixel_values":   pixel_values,
                "position_ids":   position_ids,
                "labels":         labels,
                "idx":            idx
            }
    return CoTDataset()

class MyCollator:
    def __init__(self, processor, label_pad_token_id=-100):
        self.pad_token_id       = processor.tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
    def __call__(self, batch):
        keys = ["input_ids", "attention_mask", "position_ids", "labels"]
        output = {}
        for key in keys:
            seqs = [ex[key] for ex in batch]
            pad_val = (self.pad_token_id if key != "labels" else self.label_pad_token_id)
            output[key] = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=pad_val
            )
        output["pixel_values"] = torch.stack([ex["pixel_values"] for ex in batch if ex["pixel_values"] is not None]) if batch[0]["pixel_values"] is not None else None
        output["idx"] = torch.tensor([ex["idx"] for ex in batch], dtype=torch.long)
        return output

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
    configs.latent_dim = int(getattr(configs, "latent_dim", 4096))
    configs.n_latents = int(getattr(configs, "n_latents", 8))
    configs.latent_lr = float(getattr(configs, "latent_lr", 5e-3))
    configs.e_steps = int(getattr(configs, "e_steps", 2))

    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name + "_cot",
        config=vars(configs),
        resume=True,
        reinit=True
    )

    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = configs.save_path
    os.makedirs(save_dir, exist_ok=True)
    local_ckpt_dir = "/content/checkpoints"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    metrics_csv = os.path.join(save_dir, "metrics_cot.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "avg_tokens"])

    processor = LlavaProcessor.from_pretrained(configs.model_id)
    tokenizer = processor.tokenizer
    special_tokens_dict = {
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    model = LlavaForCausalLM.from_pretrained(configs.model_id)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(processor, label_pad_token_id=-100)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = GradScaler()

    patience = 8
    best_val = float("inf")
    patience_counter = 0
    train_losses, val_losses, accuracies, token_counts = [], [], [], []
    printed_checkpoint_reminder = False
    prev_best_ckpt = None
    for epoch in range(start_epoch, configs.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} ===")
        epoch_start = time.time()
        train_ds = get_cot_dataset(train_data, processor, max_len=getattr(configs, "max_length", 512))
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
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    labels=batch["labels"]
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss += loss.item()
            pbar.set_postfix(train_loss=loss.item())
            wandb_run.log({"train/loss_step": loss.item(), "train/epoch": epoch+1})
        avg_train = total_loss / len(loader)
        train_losses.append(avg_train)
        print(f"📉 Avg train loss: {avg_train:.4f}")
        model.eval()
        val_ds = get_cot_dataset(val_data, processor, max_len=getattr(configs, "max_length", 512))
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True
        )
        vloss, correct, tot, tokens, total_gen_tokens, num_samples, exact_matches = 0.0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc="Validating", leave=True, bar_format="{l_bar}{n_fmt}/{total_fmt} [{percentage:3.0f}%]")
            for batch in vbar:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                out = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    labels=batch["labels"]
                )
                loss = out.loss
                vloss += loss.item()
                preds = out.logits.argmax(-1)
                mask = (batch["labels"] != -100)
                correct += ((preds == batch["labels"]) & mask).sum().item()
                tot += mask.sum().item()
                tokens += ((preds != processor.tokenizer.pad_token_id) & mask).sum().item()
                # --- Token counting logic for CoT ---
                pred_answer_ids = preds[0].tolist()
                output_text = processor.tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
                num_gen_tokens = len(processor.tokenizer.encode(output_text, add_special_tokens=False))
                total_gen_tokens += num_gen_tokens
                num_samples += 1
                # --- Answer-level accuracy ---
                idxs = batch["idx"]
                gt_answer = val_data[idxs[0].item()]["answer"]
                if output_text.strip().lower() == gt_answer.strip().lower():
                    exact_matches += 1
                vbar.set_postfix(val_loss=loss.item())
        avg_vl = vloss / len(val_loader)
        acc = 100 * correct / tot
        avg_tk = tokens / len(val_loader)
        avg_gen_tokens = total_gen_tokens / num_samples if num_samples > 0 else 0
        answer_acc = 100 * exact_matches / num_samples if num_samples > 0 else 0
        val_losses.append(avg_vl)
        accuracies.append(acc)
        token_counts.append(avg_tk)
        print(f"✅ Val loss {avg_vl:.4f} | TokenAcc {acc:.2f}% | AnsAcc {answer_acc:.2f}% | AvgTokens {avg_tk:.1f} | AvgGenTokens {avg_gen_tokens:.1f}")
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train, avg_vl, acc, answer_acc, avg_tk, avg_gen_tokens])
        if avg_vl < best_val:
            if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
            best_val = avg_vl
            patience_counter = 0
            best_ckpt_name = f"best_epoch{epoch+1}_valloss{avg_vl:.4f}.pt"
            best_ckpt = os.path.join(local_ckpt_dir, best_ckpt_name)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch+1
            }, best_ckpt)
            best_info = {"epoch": epoch+1, "val_loss": avg_vl, "val_acc": acc, "avg_tokens": avg_tk}
            with open(os.path.join(save_dir, "best_info_cot.json"), "w") as f:
                json.dump(best_info, f, indent=2)
            with open(os.path.join(local_ckpt_dir, "best_info_cot.json"), "w") as f:
                json.dump(best_info, f, indent=2)
            prev_best_ckpt = best_ckpt
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping triggered! No improvement for {patience} epochs.")
                break
        ckpt_name = f"checkpoint_epoch{epoch+1}_valloss{avg_vl:.4f}.pt"
        ckpt = os.path.join(local_ckpt_dir, ckpt_name)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch+1
        }, ckpt)
        wandb.save(ckpt, base_path=local_ckpt_dir)
        if not printed_checkpoint_reminder:
            print(f"[INFO] To keep a specific checkpoint (e.g., {ckpt_name}), copy it to Drive before your Colab session ends:")
            print(f"!cp /content/checkpoints/{ckpt_name} {save_dir}/")
            printed_checkpoint_reminder = True
        fig = plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()
        wandb.log({f"loss_plot_{epoch+1}": wandb.Image(fig)})
        plt.close(fig)
        print(f"⏱ Epoch time: {(time.time() - epoch_start)/60:.2f} mins")
    wandb_run.finish()

if __name__ == "__main__":
    main() 