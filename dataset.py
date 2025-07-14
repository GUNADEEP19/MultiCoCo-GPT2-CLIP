import json
import random
import torch
from torch.utils.data import Dataset
from transformers import LlavaProcessor
import itertools
from PIL import Image  # <-- Add this import
import os

def get_dataset(path, tokenizer=None, max_size=1_000_000_00):
    with open(path, "r") as f:
        raw = json.load(f)[:max_size]
    return [
        {
            "question": d["question"],
            "steps": d["steps"],
            "answer": d["answer"],  # already string
            "image": d.get("image", None),
            "idx": i
        }
        for i, d in enumerate(raw)
    ]

def get_cot_latent_dataset(base_dataset, scheduled_stage, configs,
                           start_id, latent_id, end_id,
                           no_special_marker=False, shuffle=False):
    processor = LlavaProcessor.from_pretrained(configs.model_id)
    c_thought        = configs.c_thought
    uniform_prob     = configs.uniform_prob
    max_len          = getattr(configs, "max_length", 1024)

    class StageDataset(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            total_steps = len(ex["steps"]) if ex.get("steps") else 0
            n_latent = min(getattr(configs, "n_latents", 10), total_steps)
            remaining_steps = ex["steps"][n_latent:] if total_steps > n_latent else []
            q = ex["question"]
            s = ex["steps"]
            a = ex["answer"]
            # Compose text with latent tokens
            question_with_latents = q
            if not no_special_marker:
                question_with_latents += " <|start-latent|>"
            question_with_latents += " " + "<|latent|> " * n_latent
            max_latents = getattr(configs, "n_latents", 10)
            if n_latent < max_latents:
                question_with_latents += "<|latent|> " * (max_latents - n_latent)
            if not no_special_marker:
                question_with_latents += "<|end-latent|> "
            question_with_latents += " " + " ".join(remaining_steps)
            question_with_latents += " " + a
            # Build LLaVA chat template conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question_with_latents.strip()}
                    ]
                }
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            image_path = ex.get("image", None)
            if image_path is not None:
                if not os.path.isabs(image_path):
                    pass
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = processor.image_processor(img, return_tensors="pt")["pixel_values"][0]
                except Exception as e:
                    print(f"[WARNING] Could not load image {image_path}: {e}")
                    img_tensor = None
            else:
                img_tensor = None
            processed = processor(text=prompt, images=img_tensor, return_tensors="pt", padding="max_length", max_length=max_len)
            input_ids = processed.input_ids[0]
            attention_mask = processed.attention_mask[0]
            pixel_values = processed.pixel_values[0] if img_tensor is not None else None
            position_ids = torch.arange(len(input_ids)).clamp(max=max_len-1)
            # Labels: mask out everything before answer
            label_offset = (input_ids == processor.tokenizer.convert_tokens_to_ids("<|latent|>")).nonzero(as_tuple=True)[0].max().item() + 1 if n_latent > 0 else len(q)
            labels = input_ids.clone()
            labels[:label_offset] = -100
            # Also mask <|end-latent|> token(s)
            end_latent_id = processor.tokenizer.convert_tokens_to_ids("<|end-latent|>")
            end_latent_positions = (input_ids == end_latent_id).nonzero(as_tuple=True)[0]
            labels[end_latent_positions] = -100
            # Mask unused latent slots (if any)
            latent_id = processor.tokenizer.convert_tokens_to_ids("<|latent|>")
            latent_positions = (input_ids == latent_id).nonzero(as_tuple=True)[0]
            if len(latent_positions) > n_latent:
                mask_out = latent_positions[n_latent:]
                labels[mask_out] = -100
            return {
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "pixel_values":   pixel_values,
                "position_ids":   position_ids,
                "labels":         labels,
                "idx":            idx
            }
    ds = StageDataset()
    if shuffle:
        perm = torch.randperm(len(ds)).tolist()
        return torch.utils.data.Subset(ds, perm)
    return ds

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
