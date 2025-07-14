import os
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
# LLaVA imports
from llava_hf import LlavaForCausalLM, LlavaProcessor

# Suppress specific warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*Was asked to gather along dimension 0.*")

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

class Coconut(nn.Module):
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

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        pixel_values=None,
        position_ids=None,
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

        # LLaVA expects pixel_values, input_ids, attention_mask, position_ids, labels
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
        # For compatibility with old code
        return Outputs(loss=loss, inputs_embeds=None, logits=logits)

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
        assert input_ids.size(0) == 1, "Only batch_size=1 supported"
        tokens = input_ids[0].tolist()
        labels = input_ids.clone()
        out = self.forward(input_ids, attention_mask, labels, pixel_values=pixel_values)
        logits = out.logits
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)
        for _ in range(max_new_tokens - 1):
            new_input_ids = torch.tensor([tokens], device=input_ids.device)
            out2 = self.base_causallm(input_ids=new_input_ids, pixel_values=pixel_values)
            next_token = out2.logits[0, -1].argmax().item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            self.gen_forward_cnt += 1
        out_ids = torch.tensor(tokens, device=input_ids.device).unsqueeze(0)
        return out_ids
