# coconut.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import CLIPModel

# ─── Suppress specific deprecation/user warnings ────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # get embeddings
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        # CLIP vision encoder + projector
        self.vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_projector = nn.Linear(
            self.vision_encoder.config.projection_dim,
            self.embedding.embedding_dim,
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        pixel_values=None,
        **kwargs
    ):
        device = self.embedding.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        position_ids = position_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # ✅ Safety device assertions
        assert input_ids.device == device
        assert attention_mask.device == device
        assert position_ids.device == device
        assert labels.device == device

        logits = []

        # find latent positions per example
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == batch_i]
            for batch_i in range(input_ids.shape[0])
        ]
        max_n_latents = max(len(l) for l in latent_lists)
        next_compute_range = (0, input_ids.shape[1])

        # embed tokens
        inputs_embeds = self.embedding(input_ids)

        # inject image features
        if pixel_values is not None:
            with torch.no_grad():
                image_features = self.vision_encoder.get_image_features(
                    pixel_values=pixel_values
                )
            image_features = self.vision_projector(image_features)
            # add to first token embed
            inputs_embeds[:, 0, :] += image_features

        # if there are latents, compute up to the first latent
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # stage‑wise passes
        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # ─── DEBUG: Check device mismatch ───────────────────────────────
                print(f"[DEBUG] input_ids.device       = {input_ids.device}")
                print(f"[DEBUG] position_ids.device    = {position_ids.device}")
                print(f"[DEBUG] attention_mask.device  = {attention_mask.device}")
                print(f"[DEBUG] inputs_embeds.device   = {inputs_embeds.device}")
                # ────────────────────────────────────────────────────────────────
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                offset = 0
            else:
                past = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    past_key_values=past,
                    output_hidden_states=True,
                )
                offset = next_compute_range[0]

            logits.append(outputs.logits)
            next_compute_range = (
                next_compute_range[1],
                input_ids.shape[1]
                if pass_idx + 1 >= max_n_latents
                else next_compute_range[1] + 1,
            )
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            for b_idx, latent_pos_list in enumerate(latent_lists):
                if pass_idx < len(latent_pos_list):
                    t_idx = latent_pos_list[pass_idx]
                    local_idx = t_idx - 1 - offset
                    if 0 <= local_idx < hidden_states.shape[1]:
                        inputs_embeds[b_idx, t_idx, :] = hidden_states[b_idx, local_idx, :]

        final_past = (
            [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
            if kv_cache else None
        )
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0]:next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[
                :, next_compute_range[0]:next_compute_range[1]
            ],
            past_key_values=final_past,
            output_hidden_states=True,
        )
        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1
        logits = torch.cat(logits, dim=-2)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="mean")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        assert input_ids.shape[0] == 1, "Only batch_size=1 is supported"
        tokens = input_ids[0].tolist()
        labels = input_ids.clone()
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).view(1, -1)

        out = self.forward(
            input_ids,
            attention_mask,
            labels,
            position_ids,
        )
        inputs_embeds = out.inputs_embeds
        next_token = out.logits[0, -1].argmax().item()
        tokens.append(next_token)
        new_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        seq_embeds = torch.cat((inputs_embeds, new_embed), dim=1)

        for _ in range(max_new_tokens - 1):
            out2 = self.base_causallm(inputs_embeds=seq_embeds)
            next_token = out2.logits[0, -1].argmax().item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            seq_embeds = torch.cat((seq_embeds, new_embed), dim=1)
            self.gen_forward_cnt += 1

        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                _ = self.base_causallm(inputs_embeds=seq_embeds)
                self.gen_forward_cnt += 1

        out_ids = torch.tensor(tokens, device=input_ids.device).view(1, -1)
        return (out_ids, seq_embeds) if output_embedding else out_ids
