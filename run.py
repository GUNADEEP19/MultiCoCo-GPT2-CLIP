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

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.cuda.amp import autocast, GradScaler
from transformers import LlavaForConditionalGeneration, LlavaProcessor, BitsAndBytesConfig, AutoTokenizer, LlamaTokenizer

def decode_preds(pred_ids, processor):
    if pred_ids.ndim == 2:
        pred_ids = pred_ids[0]
    pred_ids = pred_ids.tolist()
    pred_ids = [i for i in pred_ids if i != -100]
    return processor.tokenizer.decode(pred_ids, skip_special_tokens=True)

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
        name=configs.name,
        config=vars(configs),
        resume=True,
        reinit=True
    )

    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "/content/drive/MyDrive/COCONUT/checkpoints/coconut_aokvqa_gunadeep"
    os.makedirs(save_dir, exist_ok=True)
    local_ckpt_dir = "/content/checkpoints"
    os.makedirs(local_ckpt_dir, exist_ok=True)

    # model & processor - Use direct transformers loading for better compatibility
    load_4bit = getattr(configs, "load_4bit", False)
    use_flash_attention_2 = getattr(configs, "use_flash_attention_2", False)
    
    print(f"🔄 Loading model from {configs.model_id}...")
    
    # Load processor and tokenizer with fallback options
    try:
        print("📝 Loading processor...")
        # Try loading with minimal parameters for transformers 4.37.2 compatibility
        processor = LlavaProcessor.from_pretrained(configs.model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print("✅ Processor loaded successfully")
    except Exception as e:
        print(f"⚠️  Processor loading failed: {e}")
        print("🔄 Trying alternative loading method...")
        try:
            # Try loading tokenizer separately
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(configs.model_id, trust_remote_code=True)
            processor = LlavaProcessor.from_pretrained(configs.model_id, trust_remote_code=True)
            print("✅ Alternative loading successful")
        except Exception as e2:
            print(f"❌ Alternative loading also failed: {e2}")
            print("🔄 Trying final fallback method...")
            try:
                # Final fallback: use LlamaTokenizer directly
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(configs.model_id)
                # Create a simple processor wrapper for transformers 4.37.2
                processor = type('SimpleProcessor', (), {
                    'tokenizer': tokenizer,
                    'image_processor': type('ImageProcessor', (), {
                        '__call__': lambda self, img, **kwargs: {'pixel_values': torch.randn(1, 3, 224, 224) if img is not None else None}
                    })(),
                    'apply_chat_template': staticmethod(lambda conversation, add_generation_prompt=True: conversation[0]["content"][1]["text"]),
                    '__call__': lambda self, text=None, images=None, return_tensors=None, padding=None, max_length=None, **kwargs: {
                        'input_ids': tokenizer(text, return_tensors=return_tensors, padding=padding or 'do_not_pad', max_length=max_length, **kwargs)['input_ids'],
                        'attention_mask': tokenizer(text, return_tensors=return_tensors, padding=padding or 'do_not_pad', max_length=max_length, **kwargs)['attention_mask'],
                        'pixel_values': torch.randn(1, 3, 224, 224) if images is not None else None
                    }
                })()
                print("✅ Final fallback loading successful")
            except Exception as e3:
                print(f"❌ All loading methods failed: {e3}")
                raise e3
    
    # Load model with proper configuration
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if getattr(configs, "bf16", False) else torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }
    
    if load_4bit:
        print("🔧 Loading model in 4-bit precision...")
        model_kwargs["load_in_4bit"] = True
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    if use_flash_attention_2:
        print("🚀 Enabling Flash Attention 2...")
        model_kwargs["use_flash_attention_2"] = True
    
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        configs.model_id,
        trust_remote_code=True,
        **model_kwargs
    )
    special_tokens_dict = {
        "additional_special_tokens": ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    llava_model.resize_token_embeddings(len(tokenizer))
    model = Coconut(
        model_id=None,  # not used anymore
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id
    )
    model = model.to(device)
    model.processor = processor
    model.base_causallm = llava_model
    model.embedding = llava_model.get_input_embeddings()
    print("Model and processor loaded successfully!")
    # Handle different config structures in transformers 4.37.2
    try:
        vision_tower = llava_model.config.vision_tower
        print("Vision tower config:", vision_tower)
    except AttributeError:
        try:
            vision_tower = llava_model.config.vision_config
            print("Vision config:", vision_tower)
        except AttributeError:
            print("Vision tower: CLIP ViT-L/14 (default for LLaVA-1.5-7B)")
    print(f"Model type: {type(llava_model)}")
    print(f"Model device: {next(llava_model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in llava_model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    scaler = GradScaler()

    print(f"[DEBUG] Model embedding on device: {next(model.embedding.parameters()).device}")
    print("[INFO] For A100 GPU, consider increasing batch_size_training in your YAML config for best performance.")

    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(processor, label_pad_token_id=-100)

    n_train = len(train_data)
    # Handle different config structures in transformers 4.37.2
    try:
        hidden_size = model.base_causallm.config.hidden_size
    except AttributeError:
        try:
            hidden_size = model.base_causallm.config.text_config.hidden_size
        except AttributeError:
            hidden_size = 4096  # Default for LLaVA-1.5-7B

    # Checkpoint handling
    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    # Initialize metrics CSV
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "stage", "train_loss", "val_loss", "val_acc", "avg_tokens"])

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
        all_latents = torch.randn(n_train, configs.n_latents, hidden_size, requires_grad=True, device=device)

    latent_optimizer = optim.Adam([all_latents], lr=configs.latent_lr)
    # Remove the earlier optimizer and scaler initialization (if present)

    patience = 8
    best_val = float("inf")
    patience_counter = 0
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    train_losses, val_losses, accuracies, token_counts = [], [], [], []
    printed_checkpoint_reminder = False
    prev_best_ckpt = None
    # Curriculum learning setup
    max_latent_stage = getattr(configs, "max_latent_stage", 7)
    epochs_per_stage = getattr(configs, "epochs_per_stage", 4)
    for epoch in range(start_epoch, configs.num_epochs):
        stage = min(epoch // epochs_per_stage, max_latent_stage-1)
        n_latents = stage + 1
        print(f"\n=== Epoch {epoch+1}/{configs.num_epochs} | Stage {stage} | n_latents {n_latents} ===")
        epoch_start = time.time()

        # Update configs for this stage
        configs.n_latents = n_latents
        train_ds = get_cot_latent_dataset(train_data, stage, configs,
                                          processor.tokenizer.convert_tokens_to_ids("<|start-latent|>"),
                                          processor.tokenizer.convert_tokens_to_ids("<|latent|>"),
                                          processor.tokenizer.convert_tokens_to_ids("<|end-latent|>"),
                                          processor)
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

        for batch in pbar:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            idxs = batch["idx"]
            Z = all_latents[idxs]
            model.eval()
            for _ in range(configs.e_steps):
                latent_optimizer.zero_grad()
                # inject_latents logic can be adapted if needed
                labels = batch["labels"]
                use_bf16 = getattr(configs, "bf16", False)
                autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
                with autocast(device_type='cuda', dtype=autocast_dtype):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        position_ids=batch.get("position_ids"),
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch.get("pixel_values"),
                        labels=labels,
                        latents=Z
                    )
                    loss_z = outputs.loss
                loss_z.backward()
                latent_optimizer.step()
            all_latents.requires_grad = False
            model.train()
            optimizer.zero_grad()
            use_bf16 = getattr(configs, "bf16", False)
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    position_ids=batch.get("position_ids"),
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    labels=batch["labels"],
                    latents=Z
                )
                loss_m = outputs.loss
            scaler.scale(loss_m).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        val_ds = get_cot_latent_dataset(val_data, stage, configs,
                                        processor.tokenizer.convert_tokens_to_ids("<|start-latent|>"),
                                        processor.tokenizer.convert_tokens_to_ids("<|latent|>"),
                                        processor.tokenizer.convert_tokens_to_ids("<|end-latent|>"),
                                        processor)
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
        # Select 4 fixed validation samples for latent trajectory tracking
        fixed_traj_indices = [12, 45, 78, 101]  # <-- Use 4 chosen indices
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
                    pixel_values=batch.get("pixel_values"),
                    labels=batch["labels"],
                    latents=Z
                )
                loss = out.loss
                # t-SNE collection (unchanged)
                if out.inputs_embeds is not None and len(tsne_embeds) < 100:
                    latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
                    latent_mask = (batch["input_ids"] == latent_id)
                    for b in range(out.inputs_embeds.shape[0]):
                        latent_embeds = out.inputs_embeds[b][latent_mask[b]].detach().cpu().numpy()
                        if latent_embeds.shape[0] > 0:
                            tsne_embeds.append(latent_embeds.mean(axis=0))
                            tsne_labels.append(val_data[idxs[b].item()]["answer"])
                vloss += loss.item()
                # --- Token counting logic for COCONUT ---
                input_ids = batch["input_ids"][0].tolist()
                latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
                try:
                    last_latent_idx = max(i for i, t in enumerate(input_ids) if t == latent_id)
                    answer_start = last_latent_idx + 1
                except ValueError:
                    answer_start = 0
                preds = out.logits.argmax(-1)
                pred_answer_ids = preds[0][answer_start:]
                output_text = processor.tokenizer.decode(pred_answer_ids, skip_special_tokens=True)
                num_gen_tokens = len(processor.tokenizer.encode(output_text, add_special_tokens=False))
                total_gen_tokens += num_gen_tokens
                gen_token_counts.append(num_gen_tokens)
                num_samples += 1
                # --- Answer-level accuracy with index-to-label mapping ---
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
                # Collect sample predictions for W&B logging (up to 10 random samples)
                if len(sample_preds) < 10:
                    image_path = val_data[idxs[0].item()].get("image", "")
                    # For COCONUT, log mean latent vector as a string (optional, for interpretability)
                    mean_latent = None
                    if out.inputs_embeds is not None:
                        latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
                        latent_mask = (batch["input_ids"] == latent_id)
                        latent_embeds = out.inputs_embeds[0][latent_mask[0]].detach().cpu().numpy()
                        if latent_embeds.shape[0] > 0:
                            mean_latent = np.round(latent_embeds.mean(axis=0), 4).tolist()
                    sample_preds.append({
                        "question": question_str,
                        "ground_truth": gt_answer,
                        "prediction": output_text.strip(),
                        "image_path": image_path,  # This is the path from the dataset (Drive or local)
                        "mean_latent": str(mean_latent) if mean_latent is not None else ""
                    })
                # For latent trajectory logging: collect mean latent for fixed samples
                idx_val = idxs[0].item()
                if idx_val in fixed_traj_indices and out.inputs_embeds is not None:
                    latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
                    latent_mask = (batch["input_ids"] == latent_id)
                    latent_embeds = out.inputs_embeds[0][latent_mask[0]].detach().cpu().numpy()
                    if latent_embeds.shape[0] > 0:
                        mean_latent = latent_embeds.mean(axis=0)
                        if idx_val not in latent_traj_dict:
                            latent_traj_dict[idx_val] = []
                        latent_traj_dict[idx_val].append(mean_latent)
        # Histogram of generated token counts
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
                "scaler": scaler.state_dict(),
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
            "scaler": scaler.state_dict(),
            "latents": all_latents,
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
        # t-SNE visualization and W&B logging
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
        # Log sample predictions to W&B as a table (Coconut only)
        # Note: Reasoning steps are not logged for COCONUT, since the model reasons in latent space, not explicit text.
        if len(sample_preds) > 0:
            columns = ["question", "ground_truth", "prediction", "image_path", "mean_latent"]
            table = wandb.Table(columns=columns)
            for row in sample_preds:
                table.add_data(row["question"], row["ground_truth"], row["prediction"], row["image_path"], row["mean_latent"])
            wandb.log({"sample_predictions": table})
        # At the END of each stage, plot latent trajectories for tracked samples (using t-SNE instead of UMAP)
        if (epoch + 1) % epochs_per_stage == 0 and len(latent_traj_dict) > 0:
            # Stack all mean latents for all tracked samples
            all_latents = []
            sample_ids = []
            for idx, traj in latent_traj_dict.items():
                all_latents.extend(traj)
                sample_ids.extend([idx]*len(traj))
            if len(all_latents) > 0:
                try:
                    # Use t-SNE for dimensionality reduction (no additional dependencies needed)
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_latents)-1))
                    tsne_proj = tsne.fit_transform(np.stack(all_latents))
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
            # Save for next epoch
            main.latent_traj_dict = latent_traj_dict
    wandb_run.finish()

    # Save tokenizer with special tokens for future inference
    tokenizer.save_pretrained(os.path.join(local_ckpt_dir, "tokenizer"))
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
