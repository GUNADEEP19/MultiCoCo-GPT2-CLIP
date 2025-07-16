import os
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple

# Suppress specific warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class CoconutCoT(nn.Module):
    def __init__(self, gpt2, clip, eos_token_id):
        super().__init__()
        self.gpt2 = gpt2
        self.clip = clip
        self.eos_token_id = eos_token_id
        self.embedding = gpt2.get_input_embeddings()
        # Project CLIP embedding to GPT-2 hidden size
        self.img_proj = nn.Linear(clip.visual.output_dim, gpt2.config.hidden_size)

    def forward(self, input_ids, attention_mask, labels, img_embeds=None, position_ids=None, **kwargs):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        input_embeds = self.embedding(input_ids)
        # Prepend image embedding as first token if provided
        if img_embeds is not None:
            img_embeds = img_embeds.to(device)
            input_embeds = torch.cat([img_embeds.unsqueeze(1), input_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
            labels = torch.cat([
                torch.full((labels.shape[0], 1), -100, device=device, dtype=labels.dtype),
                labels
            ], dim=1)
        outputs = self.gpt2(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs
        )
        logits = outputs.logits
        loss = outputs.loss
        return Outputs(loss=loss, inputs_embeds=input_embeds, logits=logits)

    def train(self, mode: bool = True):
        self.gpt2.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(self, input_ids, attention_mask, img_embeds=None, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
        assert input_ids.size(0) == 1, "Only batch_size=1 supported"
        input_embeds = self.embedding(input_ids.to(self.embedding.weight.device))
        if img_embeds is not None:
            img_embeds = img_embeds.to(self.embedding.weight.device)
            input_embeds = torch.cat([img_embeds.unsqueeze(1), input_embeds], dim=1)
            attention_mask = torch.cat([
                torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
        gen_kwargs = dict(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id
        )
        gen_kwargs.update(kwargs)
        out_ids = self.gpt2.generate(**gen_kwargs)
        return out_ids 