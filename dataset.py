import json
import random
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import itertools

def get_dataset(path, tokenizer=None, max_size=1_000_000_00):
    with open(path, "r") as f:
        raw = json.load(f)[:max_size]
    return [
        {
            "question": d["question"],
            "steps": d["steps"],
            "answer": d["answer"],  # already string
            "idx": i
        }
        for i, d in enumerate(raw)
    ]

def get_cot_latent_dataset(base_dataset, scheduled_stage, configs,
                           start_id, latent_id, end_id,
                           no_special_marker=False, shuffle=False):
    tokenizer = GPT2Tokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    c_thought        = configs.c_thought
    max_latent_stage = configs.max_latent_stage
    uniform_prob     = configs.uniform_prob
    max_len          = getattr(configs, "max_length", 1024)

    class StageDataset(Dataset):
        def __len__(self): return len(base_dataset)
        def __getitem__(self, idx):
            ex = base_dataset[idx]
            # choose how many steps to hide vs latent
            if random.random() < uniform_prob:
                k = random.randint(0, len(ex["steps"]))
            else:
                k = scheduled_stage
            k = min(k, max_latent_stage)
            n_latent = k * c_thought

            # tokenize
            q_tokens = tokenizer.encode(ex["question"], add_special_tokens=True)
            s_tokens = [tokenizer.encode(s + "\n", add_special_tokens=False)
                        for s in ex["steps"]]
            a_tokens = tokenizer.encode(ex["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]

            tokens = q_tokens.copy()
            if not no_special_marker:
                tokens.append(start_id)
            tokens.extend([latent_id] * n_latent)
            if not no_special_marker:
                tokens.append(end_id)
            tokens.extend(itertools.chain.from_iterable(s_tokens[k:]))
            tokens.extend(a_tokens)

            tokens = tokens[:max_len]
            attention_mask = [1] * len(tokens)
            position_ids = [min(i, 1023) for i in range(len(tokens))]

            label_offset = len(q_tokens) + n_latent + (0 if no_special_marker else 2)
            labels = [-100] * label_offset + tokens[label_offset:]
            labels = labels[:len(tokens)]
            while len(labels) < len(tokens):
                labels.append(-100)

            return {
                "input_ids":      torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "position_ids":   torch.tensor(position_ids, dtype=torch.long),
                "labels":         torch.tensor(labels, dtype=torch.long),
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
    def __init__(self, tokenizer, latent_id, label_pad_token_id=-100):
        self.pad_token_id       = tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        keys = ["input_ids", "attention_mask", "position_ids", "labels"]
        output = {}
        for key in keys:
            seqs = [ex[key] for ex in batch]
            if key == "position_ids":
                pad_val = 1023
            else:
                pad_val = (self.pad_token_id if key != "labels" else self.label_pad_token_id)
            output[key] = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=pad_val
            )
        output["idx"] = torch.tensor([ex["idx"] for ex in batch], dtype=torch.long)
        return output
