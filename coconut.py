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
MAX_N_LATENT = 8  # For gpt2-xl, hidden size is 1600

class Coconut(nn.Module):
    def __init__(self, gpt2, clip, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__()
        self.gpt2 = gpt2
        self.clip = clip
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.embedding = gpt2.get_input_embeddings()
        # Project CLIP embedding to GPT-2 hidden size (gpt2-xl: 1600)
        self.img_proj = nn.Linear(clip.visual.output_dim, gpt2.config.hidden_size)

    def inject_latents(self, input_ids, latents, img_embeds=None):
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        input_embeds = self.embedding(input_ids.to(device))
        batch_size, seq_len = input_ids.shape
        latent_token_id = self.latent_token_id
        latents = latents.to(device=device, dtype=dtype)
        for b in range(batch_size):
            latent_positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0]
            n_lat = min(len(latent_positions), latents.shape[1])
            if n_lat > 0:
                input_embeds[b, latent_positions[:n_lat], :] = latents[b, :n_lat, :]
        # Prepend image embedding as first token
        if img_embeds is not None:
            img_embeds = img_embeds.to(device=device, dtype=dtype)
            input_embeds = torch.cat([img_embeds.unsqueeze(1), input_embeds], dim=1)
        return input_embeds

    def forward(self, input_ids, attention_mask, labels, img_embeds=None, position_ids=None, latents=None, **kwargs):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        if latents is not None:
            input_embeds = self.inject_latents(input_ids, latents, img_embeds)
            # Adjust attention mask and labels for prepended image token
            attention_mask = torch.cat([torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype), attention_mask], dim=1)
            labels = torch.cat([torch.full((labels.shape[0], 1), -100, device=device, dtype=labels.dtype), labels], dim=1)
            outputs = self.gpt2(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                **kwargs
            )
        else:
            input_embeds = None
            outputs = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                **kwargs
            )
        logits = outputs.logits
        loss = outputs.loss
        return Outputs(loss=loss, inputs_embeds=input_embeds if latents is not None else None, logits=logits)

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
            attention_mask = torch.cat([torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype), attention_mask], dim=1)
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

# === EMA helper ===
def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay) 