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
        """
        Continuous-thought forward pass.

        - When sequence contains <|latent|> tokens, we run a streaming forward pass using
          past_key_values and replace the current token's embedding with the previous hidden
          state for these latent positions.
        - If there are no latent tokens, fall back to a single forward pass for speed.
        - If img_embeds is provided, prepend a learned image embedding step (projected CLIP
          embedding) and align attention/labels accordingly.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # If no latent tokens are present in the batch and no special streaming is required,
        # use the fast path (single forward). We still support optional image prepend.
        has_latent = (input_ids == self.latent_token_id).any().item()

        if not has_latent and latents is None:
            # Fast path
            if img_embeds is not None:
                # Build inputs_embeds with image embedding prepended
                token_embeds = self.embedding(input_ids)
                img_embeds = self.img_proj(img_embeds.to(device)).to(dtype)
                token_embeds = torch.cat([img_embeds.unsqueeze(1), token_embeds], dim=1)
                attention_mask = torch.cat([
                    torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype),
                    attention_mask
                ], dim=1)
                labels = torch.cat([
                    torch.full((labels.shape[0], 1), -100, device=device, dtype=labels.dtype),
                    labels
                ], dim=1)
                outputs = self.gpt2(
                    inputs_embeds=token_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids.to(device) if position_ids is not None else None,
                    labels=labels,
                    **kwargs
                )
            else:
                outputs = self.gpt2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids.to(device) if position_ids is not None else None,
                    labels=labels,
                    **kwargs
                )
            return Outputs(loss=outputs.loss, inputs_embeds=None, logits=outputs.logits)

        # Slow path: streaming with continuous-thought substitution
        batch_size, seq_len = input_ids.shape

        # Prepare augmented attention and labels if image embedding is provided
        use_image_token = img_embeds is not None
        if use_image_token:
            img_embeds_proj = self.img_proj(img_embeds.to(device)).to(dtype)
            # Augment labels and attention to account for the extra image step
            attention_mask = torch.cat([
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype),
                attention_mask
            ], dim=1)
            labels = torch.cat([
                torch.full((batch_size, 1), -100, device=device, dtype=labels.dtype),
                labels
            ], dim=1)

        # Mask out padded positions from contributing to loss
        # (where attention_mask == 0)
        labels = labels.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        # Streaming forward
        steps_total = seq_len + (1 if use_image_token else 0)
        logits_steps = []
        past_key_values = None
        prev_hidden = None

        # Optional initial image step
        if use_image_token:
            step_embed = img_embeds_proj.unsqueeze(1)  # (B, 1, H)
            out = self.gpt2(
                inputs_embeds=step_embed,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1, :]
            logits_steps.append(out.logits)  # (B, 1, V)

        # Iterate over tokens
        for pos in range(seq_len):
            token_ids_t = input_ids[:, pos]
            # Default token embedding
            token_embed = self.embedding(token_ids_t)
            # Replace with previous hidden state for latent tokens (except when prev_hidden is None)
            if prev_hidden is not None:
                is_latent = (token_ids_t == self.latent_token_id)
                if is_latent.any():
                    # Broadcast prev_hidden into token_embed for those positions
                    token_embed = torch.where(
                        is_latent.view(-1, 1),
                        prev_hidden,
                        token_embed,
                    )
            # Forward one step
            out = self.gpt2(
                inputs_embeds=token_embed.unsqueeze(1),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1, :]
            logits_steps.append(out.logits)  # (B, 1, V)

        # Concatenate logits across steps
        logits = torch.cat(logits_steps, dim=1)  # (B, steps_total, V)

        # Compute standard LM loss with left shift
        # Align labels to the streamed sequence length
        # labels shape already (B, steps_total) due to augmentation above when needed
        vocab_size = logits.size(-1)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # shift so that tokens < n predict token n
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=None, logits=logits)

    def train(self, mode: bool = True):
        self.gpt2.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(self, input_ids, attention_mask, img_embeds=None, max_new_tokens=16, output_embedding=False, synced_gpus=False, **kwargs):
        """
        Streaming generate that supports latent-thought conditioning in the prefix.
        Assumes the input_ids may already contain <|latent|> placeholders. We step
        through the prefix, applying continuous-thought substitution, then continue
        autoregressive decoding for max_new_tokens.
        Note: batch_size=1 restriction kept for simplicity.
        """
        assert input_ids.size(0) == 1, "Only batch_size=1 supported"
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        use_image_token = img_embeds is not None
        past_key_values = None
        prev_hidden = None
        generated_ids = []

        # Optional image step
        if use_image_token:
            img_embeds_proj = self.img_proj(img_embeds.to(device)).to(dtype)
            out = self.gpt2(
                inputs_embeds=img_embeds_proj.unsqueeze(1),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1, :]

        # Consume provided prefix with latent streaming
        seq_len = input_ids.shape[1]
        for pos in range(seq_len):
            token_id = input_ids[:, pos]
            tok_embed = self.embedding(token_id)
            if prev_hidden is not None and (token_id == self.latent_token_id).any():
                tok_embed = prev_hidden
            out = self.gpt2(
                inputs_embeds=tok_embed.unsqueeze(1),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1, :]

        # Autoregressive decoding
        cur_id = None
        for _ in range(max_new_tokens):
            # Next-token logits from last step
            logits = self.gpt2.lm_head(prev_hidden)
            next_id = torch.argmax(logits, dim=-1)
            generated_ids.append(next_id)
            if next_id.item() == self.eos_token_id:
                break
            # Forward this token
            tok_embed = self.embedding(next_id)
            out = self.gpt2(
                inputs_embeds=tok_embed.unsqueeze(1),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1, :]

        if len(generated_ids) == 0:
            return input_ids
        return torch.stack(generated_ids, dim=1)

# === EMA helper ===
def update_ema(model, ema_model, decay=0.999):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay) 