import os
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
# LLaVA imports
from transformers import LlavaForCausalLM, LlavaProcessor

# Suppress specific warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

class Coconut(nn.Module):
    # NOTE: The tokenizer should have special latent tokens added and the model's embeddings resized before training.
    def __init__(
        self,
        model_id="llava-hf/llava-1.5-7b-hf",
        latent_token_id=None,
        start_latent_id=None,
        end_latent_id=None,
        eos_token_id=None,
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.processor = LlavaProcessor.from_pretrained(model_id)
        self.base_causallm = LlavaForCausalLM.from_pretrained(model_id)
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()

    def inject_latents(self, input_ids, latents):
        """
        Replace <|latent|> token embeddings in input_ids with provided per-sample latent vectors.
        input_ids: (batch, seq_len)
        latents: (batch, n_latents, hidden_size)
        Returns: input_embeds (batch, seq_len, hidden_size)
        """
        device = self.embedding.weight.device
        input_embeds = self.embedding(input_ids.to(device))
        batch_size, seq_len = input_ids.shape
        latent_token_id = self.latent_token_id
        for b in range(batch_size):
            latent_positions = (input_ids[b] == latent_token_id).nonzero(as_tuple=True)[0]
            n_lat = min(len(latent_positions), latents.shape[1])
            if n_lat > 0:
                input_embeds[b, latent_positions[:n_lat], :] = latents[b, :n_lat, :]
        return input_embeds

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values=None,
        position_ids=None,
        latents=None,
        **kwargs,
    ):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        # If latents are provided, inject them at <|latent|> positions
        if latents is not None:
            input_embeds = self.inject_latents(input_ids, latents)
            outputs = self.base_causallm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                position_ids=position_ids,
                labels=labels,
                **kwargs
            )
        else:
            input_embeds = None
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                position_ids=position_ids,
                labels=labels,
                **kwargs
            )
        logits = outputs.logits
        loss = outputs.loss
        return Outputs(loss=loss, inputs_embeds=input_embeds if latents is not None else None, logits=logits)

    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        # Use HuggingFace's generate API for better decoding options
        # Only batch_size=1 supported for now
        assert input_ids.size(0) == 1, "Only batch_size=1 supported"
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id
        )
        gen_kwargs.update(kwargs)
        out_ids = self.base_causallm.generate(**gen_kwargs)
        return out_ids

# === EMA helper ===
def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
