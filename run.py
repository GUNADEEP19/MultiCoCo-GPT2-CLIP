import os
import yaml
import time
import json
import csv
import torch
import wandb
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import copy

# Memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.cuda.amp import autocast
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

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")

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

    print(f"🔄 Loading GPT-2 from {model_id} ...")
    gpt2_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    if load_4bit:
        print("🔧 Loading GPT-2 in 4-bit precision...")
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
    print("✅ GPT-2 loaded.")

    print(f"🔄 Loading CLIP from {clip_id} ...")
    clip = CLIPModel.from_pretrained(clip_id)
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    print("✅ CLIP loaded.")

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
    print(f"Model type: {type(gpt2)}")
    print(f"Model device: {next(gpt2.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in gpt2.parameters()):,} (gpt2-xl: 1.5B params, hidden size 1600)")

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = None

    print(f"[DEBUG] Model embedding on device: {next(model.embedding.parameters()).device}")
    print_memory_usage()

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, label_pad_token_id=-100)

    n_train = len(train_data)
    hidden_size = gpt2.config.hidden_size

    # Checkpoint handling
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "train_loss", "val_loss", "val_acc", "avg_tokens"])

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        all_latents = ckpt["latents"].to(device).detach().requires_grad_(True)
        start_epoch = ckpt["epoch"]
    else:
        all_latents = torch.randn(n_train, configs.n_latents, hidden_size, requires_grad=True, device=device)

    latent_optimizer = optim.Adam([all_latents], lr=configs.latent_lr)

    patience = 8
    best_val = float("inf")
    patience_counter = 0
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    train_losses, val_losses, accuracies, token_counts = [], [], [], []
    printed_checkpoint_reminder = False
    prev_best_ckpt = None
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
        loader = DataLoader(
            train_ds,
            batch_size=configs.batch_size_training,
            shuffle=True,
            collate_fn=collator,
            pin_memory=True
        )

        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc="Training", leave=False, bar_format="{l_bar}{n_fmt}/{total_fmt} [{percentage:3.0f}%]")
        gradient_accumulation_steps = getattr(configs, "gradient_accumulation_steps", 1)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            idxs = batch["idx"]
            Z = all_latents[idxs]
            model.eval()
            for e_step in range(configs.e_steps):
                latent_optimizer.zero_grad()
                labels = batch["labels"]
                with autocast(dtype=torch.float16):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        position_ids=batch.get("position_ids"),
                        attention_mask=batch["attention_mask"],
                        img_embeds=batch.get("img_embeds"),
                        labels=labels,
                        latents=Z
                    )
                    loss_z = outputs.loss
                loss_z.backward(retain_graph=(e_step < configs.e_steps - 1))
                latent_optimizer.step()
                cleanup_memory()
            all_latents.requires_grad = False
            model.train()
            optimizer.zero_grad()
            Z_mstep = all_latents[idxs].detach().requires_grad_(False)
            with autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch["attention_mask"],
                    img_embeds=batch.get("img_embeds"),
                    labels=batch["labels"],
                    latents=Z_mstep
                )
                loss_m = outputs.loss
            scaled_loss = loss_m / gradient_accumulation_steps
            scaled_loss.backward()
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
                        ema_param.data = ema_decay * ema_param.data + (1 - ema_decay) * param.data
                cleanup_memory()
            total_loss += loss_m.item()
            all_latents.requires_grad = True
            pbar.set_postfix(train_loss=loss_m.item())
            wandb_run.log({"train/loss_step": loss_m.item(), "train/epoch": epoch+1})
        avg_train = total_loss / len(loader)
        train_losses.append(avg_train)
        print(f"📉 Avg train loss: {avg_train:.4f}")

        model.eval()
        val_ds = get_cot_latent_dataset(val_data, stage, configs,
                                        tokenizer.convert_tokens_to_ids("<|start-latent|>"),
                                        tokenizer.convert_tokens_to_ids("<|latent|>"),
                                        tokenizer.convert_tokens_to_ids("<|end-latent|>"),
                                        tokenizer,
                                        clip_processor)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True
        )
        vloss, tokens, total_gen_tokens, num_samples, exact_matches = 0.0, 0, 0, 0, 0
        tsne_embeds = []
        tsne_labels = []
        gen_token_counts = []
        sample_preds = []
        fixed_traj_indices = [12, 45, 78, 101]
        if epoch == 0:
            wandb.config.update({"latent_traj_indices": fixed_traj_indices}, allow_val_change=True)
        else:
            fixed_traj_indices = wandb.config.get("latent_traj_indices", [])
        latent_traj_dict = getattr(main, "latent_traj_dict", {}) if hasattr(main, "latent_traj_dict") else {i: [] for i in fixed_traj_indices}
        with torch.no_grad():
            vbar = tqdm(val_loader, desc="Validating", leave=False, bar_format="{l_bar}{n_fmt}/{total_fmt} [{percentage:3.0f}%]")
            for batch in vbar:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                idxs = batch["idx"]
                Z = all_latents[idxs]
                out = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch["attention_mask"],
                    img_embeds=batch.get("img_embeds"),
                    labels=batch["labels"],
                    latents=Z
                )
                loss = out.loss
                if out.inputs_embeds is not None and len(tsne_embeds) < 100:
                    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                    latent_mask = (batch["input_ids"] == latent_id)
                    for b in range(out.inputs_embeds.shape[0]):
                        latent_embeds = out.inputs_embeds[b][latent_mask[b]].detach().cpu().numpy()
                        if latent_embeds.shape[0] > 0:
                            tsne_embeds.append(latent_embeds.mean(axis=0))
                            tsne_labels.append(val_data[idxs[b].item()]["answer"])
                vloss += loss.item()
                input_ids = batch["input_ids"][0].tolist()
                latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                try:
                    last_latent_idx = max(i for i, t in enumerate(input_ids) if t == latent_id)
                    answer_start = last_latent_idx + 1
                except ValueError:
                    answer_start = 0
                preds = out.logits.argmax(-1)
                pred_answer_ids = preds[0][answer_start:]
                output_text = decode_preds(pred_answer_ids, tokenizer)
                num_gen_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
                total_gen_tokens += num_gen_tokens
                gen_token_counts.append(num_gen_tokens)
                num_samples += 1
                question_str = val_data[idxs[0].item()]["question"]
                gt_index = int(val_data[idxs[0].item()]["answer"])
                if "The choices are" in question_str:
                    choices_str = question_str.split("The choices are")[-1].strip()
                    choice_map = {}
                    for part in choices_str.split(","):
                        if ":" in part:
                            idx, val = part.strip().split(":", 1)
                            choice_map[int(idx.strip())] = val.strip()
                    gt_answer = choice_map.get(gt_index, str(gt_index))
                else:
                    gt_answer = str(gt_index)
                print(f"[PRED] Q: {question_str}")
                print(f"[PRED] GT: {gt_answer} | Pred: {output_text.strip()}")
                def normalize(text):
                    return text.lower().strip().rstrip(".,!?")
                if normalize(output_text) == normalize(gt_answer):
                    exact_matches += 1
                vbar.set_postfix(val_loss=loss.item())
                if len(sample_preds) < 10:
                    image_path = val_data[idxs[0].item()].get("image", "")
                    mean_latent = None
                    if out.inputs_embeds is not None:
                        latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                        latent_mask = (batch["input_ids"] == latent_id)
                        latent_embeds = out.inputs_embeds[0][latent_mask[0]].detach().cpu().numpy()
                        if latent_embeds.shape[0] > 0:
                            mean_latent = np.round(latent_embeds.mean(axis=0), 4).tolist()
                    sample_preds.append({
                        "question": question_str,
                        "ground_truth": gt_answer,
                        "prediction": output_text.strip(),
                        "image_path": image_path,
                        "mean_latent": str(mean_latent) if mean_latent is not None else ""
                    })
                idx_val = idxs[0].item()
                if idx_val in fixed_traj_indices and out.inputs_embeds is not None:
                    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                    latent_mask = (batch["input_ids"] == latent_id)
                    latent_embeds = out.inputs_embeds[0][latent_mask[0]].detach().cpu().numpy()
                    if latent_embeds.shape[0] > 0:
                        mean_latent = latent_embeds.mean(axis=0)
                        if idx_val not in latent_traj_dict:
                            latent_traj_dict[idx_val] = []
                        latent_traj_dict[idx_val].append(mean_latent)
        if len(gen_token_counts) > 0:
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(gen_token_counts, bins=20, color='skyblue', edgecolor='black')
            ax_hist.set_title(f"Histogram of Generated Token Counts (Epoch {epoch+1})")
            ax_hist.set_xlabel("# Generated Tokens")
            ax_hist.set_ylabel("Frequency")
            wandb.log({"gen_token_hist": wandb.Image(fig_hist)})
            plt.close(fig_hist)
        avg_vl = vloss / len(val_loader)
        avg_gen_tokens = total_gen_tokens / num_samples if num_samples > 0 else 0
        answer_acc = 100 * exact_matches / num_samples if num_samples > 0 else 0
        val_losses.append(avg_vl)
        accuracies.append(answer_acc)
        token_counts.append(avg_gen_tokens)
        print(f"✅ Val loss {avg_vl:.4f} | AnsAcc {answer_acc:.2f}% | AvgGenTokens {avg_gen_tokens:.1f}")
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, stage, avg_train, avg_vl, answer_acc, avg_gen_tokens])
        if avg_vl < best_val:
            if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                os.remove(prev_best_ckpt)
            best_val = avg_vl
            patience_counter = 0
            best_ckpt_name = f"best_epoch{epoch+1}_valloss{avg_vl:.4f}.pt"
            best_ckpt = os.path.join(local_ckpt_dir, best_ckpt_name)
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "latents": all_latents,
                "epoch": epoch+1
            }, best_ckpt)
            best_info = {"epoch": epoch+1, "val_loss": avg_vl, "val_acc": answer_acc, "avg_tokens": avg_gen_tokens}
            with open(os.path.join(save_dir, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=2)
            with open(os.path.join(local_ckpt_dir, "best_info.json"), "w") as f:
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
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "latents": all_latents,
            "epoch": epoch+1
        }, ckpt)
        wandb.save(ckpt, base_path=local_ckpt_dir)
        if not printed_checkpoint_reminder:
            print(f"[INFO] To keep a specific checkpoint (e.g., {ckpt_name}), copy it to Drive before your Colab session ends:")
            print(f"!cp ./checkpoints/{ckpt_name} {save_dir}/")
            printed_checkpoint_reminder = True
        fig = plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(val_losses, label="val")
        plt.title(f"Epoch {epoch+1}")
        plt.legend()
        wandb.log({f"loss_plot_{epoch+1}": wandb.Image(fig)})
        plt.close(fig)
        print(f"⏱ Epoch time: {(time.time() - epoch_start)/60:.2f} mins")
        if len(tsne_embeds) > 5:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_proj = tsne.fit_transform(np.stack(tsne_embeds))
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=[int(l) for l in tsne_labels], cmap="tab10", alpha=0.7)
            legend1 = ax.legend(*scatter.legend_elements(), title="Answer")
            ax.add_artist(legend1)
            ax.set_title(f"t-SNE of Latent-Conditioned Embeddings (Epoch {epoch+1})")
            wandb.log({"tsne_latent_embeds": wandb.Image(fig)})
            plt.close(fig)
        if len(sample_preds) > 0:
            columns = ["question", "ground_truth", "prediction", "image_path", "mean_latent"]
            table = wandb.Table(columns=columns)
            for row in sample_preds:
                table.add_data(row["question"], row["ground_truth"], row["prediction"], row["image_path"], row["mean_latent"])
            wandb.log({"sample_predictions": table})
        if (epoch + 1) % epochs_per_stage == 0 and len(latent_traj_dict) > 0:
            all_latents_traj = []
            sample_ids = []
            for idx, traj in latent_traj_dict.items():
                all_latents_traj.extend(traj)
                sample_ids.extend([idx]*len(traj))
            if len(all_latents_traj) > 0:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents_traj)-1))
                    tsne_proj = tsne.fit_transform(np.stack(all_latents_traj))
                    fig_traj, ax_traj = plt.subplots(figsize=(7, 6))
                    color_map = plt.cm.get_cmap('tab10', len(latent_traj_dict))
                    offset = 0
                    for i, idx in enumerate(latent_traj_dict):
                        n_points = len(latent_traj_dict[idx])
                        if n_points > 1:
                            ax_traj.plot(tsne_proj[offset:offset+n_points, 0], tsne_proj[offset:offset+n_points, 1], marker='o', label=f'Sample {idx}', color=color_map(i))
                        offset += n_points
                    ax_traj.set_title(f"Latent Trajectories (Epoch {epoch+1}) - t-SNE")
                    ax_traj.legend()
                    wandb.log({"latent_trajectories": wandb.Image(fig_traj)})
                    plt.close(fig_traj)
                except Exception as e:
                    print(f"[WARNING] Could not create latent trajectory plot: {e}")
            main.latent_traj_dict = latent_traj_dict
    wandb_run.finish()
    tokenizer.save_pretrained(os.path.join(local_ckpt_dir, "tokenizer"))
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
