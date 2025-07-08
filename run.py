import os
import yaml
import torch
import wandb
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm  # 👍 Notebook-friendly tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut import Coconut
from dataset import get_cot_latent_dataset, MyCollator, get_dataset
from utils import Config, set_seed
from torch.amp import autocast, GradScaler  # AMP for mixed precision

# Fix for duplicated tqdm lines in Jupyter/Kaggle
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML config")
    args = parser.parse_args()

    # ─── Load config ─────────────────────────────────────────────────
    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    configs = Config(cfg)
    configs.lr = float(configs.lr)
    configs.weight_decay = float(configs.weight_decay)
    configs.resume = int(configs.resume)

    # ─── W&B Login & Init ────────────────────────────────────────────
    wandb.login()
    wandb_run = wandb.init(
        project=configs.project,
        name=configs.name,
        config=vars(configs),
        reinit=True
    )

    # ─── Repro & Device ───────────────────────────────────────────────
    set_seed(configs.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Prepare dirs & Resume ────────────────────────────────────────
    save_dir = os.path.join(configs.save_path, configs.name)
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = configs.resume
    ckpt_path = os.path.join(save_dir, f"checkpoint_{start_epoch}.pt") if start_epoch > 0 else None

    # ─── Load model & tokenizer ──────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Add Coconut special tokens
    special_tokens = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    LATENT_ID = tokenizer.convert_tokens_to_ids("<|latent|>")
    START_ID = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    END_ID = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    if configs.coconut:
        model = Coconut(
            base_causallm=model,
            latent_token_id=LATENT_ID,
            start_latent_id=START_ID,
            end_latent_id=END_ID,
            eos_token_id=tokenizer.eos_token_id
        )

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print("✅ Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"🔁 Resuming from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ─── Load Data ────────────────────────────────────────────────────
    train_data = get_dataset(configs.train_path)
    val_data = get_dataset(configs.val_path)
    collator = MyCollator(tokenizer, latent_id=LATENT_ID)

    optimizer = optim.AdamW(
        model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay
    )
    scaler = GradScaler("cuda")  # AMP mixed-precision

    global_step = 0
    for epoch in range(start_epoch, configs.num_epochs):
        stage = epoch // configs.epochs_per_stage
        train_ds = get_cot_latent_dataset(
            train_data, stage, configs, START_ID, LATENT_ID, END_ID
        )
        loader = DataLoader(
            train_ds,
            batch_size=configs.batch_size_training,
            shuffle=True,
            collate_fn=collator,
        )

        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{configs.num_epochs}", leave=True, ncols=100)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}

            with autocast("cuda"):
                outputs = model(**batch)
                loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                if loss.dim() != 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            pbar.set_postfix(loss=round(loss.item(), 4))

            wandb_run.log({
                "train/loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/step": global_step,
            })

        # Save checkpoint
        ckpt_epoch = epoch + 1
        ckpt_path = os.path.join(save_dir, f"checkpoint_{ckpt_epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)

    wandb_run.finish()


if __name__ == "__main__":
    main()
