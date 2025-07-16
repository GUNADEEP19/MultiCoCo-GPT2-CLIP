import json
import random
import torch
from torch.utils.data import Dataset
import itertools
from PIL import Image
import os

def get_dataset(path, tokenizer=None, max_size=1_000_000_00):
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

def get_cot_latent_dataset(base_dataset, scheduled_stage, configs,
                           start_id, latent_id, end_id,
                           tokenizer,
                           clip_processor,
                           no_special_marker=False, shuffle=False):
    c_thought        = configs.c_thought
    uniform_prob     = configs.uniform_prob
    max_len          = getattr(configs, "max_length", 256)  # gpt2-xl default

    class StageDataset(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            total_steps = len(ex["steps"]) if ex.get("steps") else 0
            n_latent = min(getattr(configs, "n_latents", 8), total_steps)  # gpt2-xl default
            remaining_steps = ex["steps"][n_latent:] if total_steps > n_latent else []
            q = ex["question"]
            s = ex["steps"]
            a = ex["answer"]
            # Compose text with latent tokens
            question_with_latents = q
            if not no_special_marker:
                question_with_latents += " <|start-latent|>"
            question_with_latents += " " + "<|latent|> " * n_latent
            max_latents = getattr(configs, "n_latents", 8)  # gpt2-xl default
            if n_latent < max_latents:
                question_with_latents += "<|latent|> " * (max_latents - n_latent)
            if not no_special_marker:
                question_with_latents += "<|end-latent|> "
            question_with_latents += " " + " ".join(remaining_steps)
            question_with_latents += " " + a
            prompt = question_with_latents.strip()
            image_path = ex.get("image", None)
            img_embeds = None
            if image_path is not None:
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = clip_processor(images=img, return_tensors="pt")["pixel_values"][0]
                    # CLIP embedding extraction will be done in the collator for batch efficiency
                except Exception as e:
                    print(f"[WARNING] Could not load image {image_path}: {e}")
                    img_tensor = None
            else:
                img_tensor = None
            processed = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_len, truncation=True)
            input_ids = processed["input_ids"][0]
            attention_mask = processed["attention_mask"][0]
            position_ids = torch.arange(len(input_ids)).clamp(max=max_len-1)
            # Labels: mask out everything before answer
            label_offset = (input_ids == tokenizer.convert_tokens_to_ids("<|latent|>")).nonzero(as_tuple=True)[0].max().item() + 1 if n_latent > 0 else len(q)
            labels = input_ids.clone()
            labels[:label_offset] = -100
            # Also mask <|end-latent|> token(s)
            end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
            end_latent_positions = (input_ids == end_latent_id).nonzero(as_tuple=True)[0]
            labels[end_latent_positions] = -100
            # Mask unused latent slots (if any)
            latent_id_ = tokenizer.convert_tokens_to_ids("<|latent|>")
            latent_positions = (input_ids == latent_id_).nonzero(as_tuple=True)[0]
            if len(latent_positions) > n_latent:
                mask_out = latent_positions[n_latent:]
                labels[mask_out] = -100
            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "img_tensor":     img_tensor,
                "position_ids":   position_ids,
                "labels":         labels,
                "idx":            idx
            }
    ds = StageDataset()
    if shuffle:
        perm = torch.randperm(len(ds)).tolist()
        return torch.utils.data.Subset(ds, perm)
    return ds

def get_cot_dataset(base_dataset, configs, tokenizer, clip_processor):
    max_len = getattr(configs, "max_length", 256)
    class CoTDataset(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            q = ex["question"]
            s = ex["steps"]
            a = ex["answer"]
            prompt = q + " " + " ".join(s) + " " + a
            image_path = ex.get("image", None)
            img_tensor = None
            if image_path is not None:
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = clip_processor(images=img, return_tensors="pt")["pixel_values"][0]
                except Exception as e:
                    print(f"[WARNING] Could not load image {image_path}: {e}")
                    img_tensor = None
            processed = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_len, truncation=True)
            input_ids = processed["input_ids"][0]
            attention_mask = processed["attention_mask"][0]
            position_ids = torch.arange(len(input_ids)).clamp(max=max_len-1)
            labels = input_ids.clone()
            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "img_tensor":     img_tensor,
                "position_ids":   position_ids,
                "labels":         labels,
                "idx":            idx
            }
    return CoTDataset()

class MyCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100, clip_model=None, device=None):
        self.pad_token_id       = tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.clip_model = clip_model
        self.device = device
    def __call__(self, batch):
        keys = ["input_ids", "attention_mask", "position_ids", "labels"]
        output = {}
        for key in keys:
            seqs = [ex[key] for ex in batch]
            pad_val = (self.pad_token_id if key != "labels" else self.label_pad_token_id)
            output[key] = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=pad_val
            )
        # Compute CLIP image embeddings in batch if possible
        img_tensors = [ex["img_tensor"] for ex in batch if ex["img_tensor"] is not None]
        if len(img_tensors) > 0 and self.clip_model is not None:
            imgs = torch.stack(img_tensors).to(self.device or "cpu")
            with torch.no_grad():
                img_embeds = self.clip_model.get_image_features(pixel_values=imgs)
            # Map back to batch positions
            img_embeds_full = []
            img_idx = 0
            for ex in batch:
                if ex["img_tensor"] is not None:
                    img_embeds_full.append(img_embeds[img_idx])
                    img_idx += 1
                else:
                    img_embeds_full.append(torch.zeros_like(img_embeds[0]))
            output["img_embeds"] = torch.stack(img_embeds_full)
        else:
            output["img_embeds"] = None
        output["idx"] = torch.tensor([ex["idx"] for ex in batch], dtype=torch.long)
        return output
