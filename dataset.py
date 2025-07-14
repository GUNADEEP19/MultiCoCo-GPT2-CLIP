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
    max_latent_stage = configs.max_latent_stage
    uniform_prob     = configs.uniform_prob
    max_len          = getattr(configs, "max_length", 1024)

    class StageDataset(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            if random.random() < uniform_prob:
                k = random.randint(0, len(ex["steps"]))
            else:
                k = scheduled_stage
            k = min(k, max_latent_stage)
            n_latent = k * c_thought
            q = ex["question"]
            s = ex["steps"]
            a = ex["answer"]
            # Compose text with latent tokens
            tokens = q
            if not no_special_marker:
                tokens += " <|start-latent|>"
            tokens += " " + "<|latent|> " * n_latent
            if not no_special_marker:
                tokens += "<|end-latent|> "
            tokens += " " + " ".join(s[k:])
            tokens += " " + a
            # Use processor to get input_ids, attention_mask, pixel_values
            image_path = ex.get("image", None)
            if image_path is not None:
                if not os.path.isabs(image_path):
                    # If the path is relative, resolve relative to the JSON file's directory or provide a base path
                    # For now, assume image_path is absolute or Colab working directory is set correctly
                    pass
                try:
                    img = Image.open(image_path).convert("RGB")
                    img_tensor = processor.image_processor(img, return_tensors="pt")["pixel_values"][0]
                except Exception as e:
                    print(f"[WARNING] Could not load image {image_path}: {e}")
                    img_tensor = None
            else:
                img_tensor = None
            processed = processor(text=tokens, images=img_tensor, return_tensors="pt", padding="max_length", max_length=max_len)
            input_ids = processed.input_ids[0]
            attention_mask = processed.attention_mask[0]
            pixel_values = processed.pixel_values[0] if img_tensor is not None else None
            position_ids = torch.arange(len(input_ids)).clamp(max=max_len-1)
            # Labels: mask out everything before answer
            label_offset = (input_ids == processor.tokenizer.convert_tokens_to_ids("<|latent|>")).nonzero(as_tuple=True)[0].max().item() + 1 if n_latent > 0 else len(q)
            labels = input_ids.clone()
            labels[:label_offset] = -100
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

def get_question_latent_dataset(scheduled_stage, base_dataset, configs,
                                start_id, latent_id, end_id,
                                no_special_marker=False):
    tokenizer = GPT2Tokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    c_thought        = configs.c_thought
    max_latent_stage = configs.max_latent_stage
    pad_to_max       = configs.pad_latent_to_max
    max_len          = getattr(configs, "max_length", 1024)

    class LatentOnly(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            q_tokens = tokenizer.encode(ex["question"], add_special_tokens=True)
            k = scheduled_stage
            if pad_to_max:
                k = max_latent_stage
            else:
                k = min(k, len(ex["steps"]))
            n_latent = k * c_thought

            tokens = q_tokens.copy()
            if not no_special_marker:
                tokens.append(start_id)
            tokens.extend([latent_id] * n_latent)
            if not no_special_marker:
                tokens.append(end_id)

            tokens = tokens[:max_len]
            attention_mask = [1] * len(tokens)
            position_ids = [min(i, 1023) for i in range(len(tokens))]

            return {
                "input_ids":      torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "position_ids":   torch.tensor(position_ids, dtype=torch.long),
                "idx":            idx
            }

    return LatentOnly()

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
