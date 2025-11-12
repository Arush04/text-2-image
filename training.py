import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.distributed import init_process_group
from torch.distributed.tensor import DeviceMesh
from torch.distributed.fsdp import fully_shard, StateDictType
from tqdm import tqdm
from PIL import Image
import random
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from modelClasses import UNET, DDPM_Scheduler
from utils import *

# --------------------------
# Training
# --------------------------
def train(
    dataset,
    tokenizer,
    text_encoder,
    batch_size=8,
    num_epochs=15,
    num_time_steps=1000,
    lr=2e-5,
    ema_decay=0.9999,
    checkpoint_path=None,
):
    # --------------------------
    # Setup
    # --------------------------
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")

    set_seed(random.randint(0, 2**32 - 1))

    # --------------------------
    # Dataset & Loader
    # --------------------------
    train_dataset = MyIterableDataset(dataset, rank, world_size, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True
    )
    print(f"Rank {rank} | Data loader ready")

    # --------------------------
    # Model & DTensor
    # --------------------------
    from torch.distributed._tensor import init_device_mesh
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data_parallel",))
    print(f"Rank {rank} | Device Mesh: {device_mesh}")

    # Create model on meta device
    # with torch.device("meta"):
    model = UNET(text_dim=512)  # adjust text_dim for your text encoder

    # Shard model
    model = fully_shard(model, mesh=device_mesh)
    # model = model.to_empty(device=torch.device("cuda"))  # materialize on GPU

    # EMA
    ema = EMA(model, decay=ema_decay)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.to("cuda")

    # Load checkpoint
    if checkpoint_path is not None and rank == 0:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["weights"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    dist.barrier()

    # Text encoder
    text_encoder = text_encoder.to("cuda")
    text_encoder.eval()

    print(f"Rank {rank} | Starting training")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for img_batch, text_batch in tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = img_batch.cuda(non_blocking=True)

            # Tokenize
            tokens = tokenizer(
                text_batch,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            )
            tokens = {k: v.to("cuda", non_blocking=True) for k, v in tokens.items()}

            with torch.no_grad():
                text_emb = text_encoder(**tokens).last_hidden_state

            t = torch.randint(0, num_time_steps, (x.shape[0],), device="cuda")
            e = torch.randn_like(x)
            a = scheduler.alpha[t].view(-1, 1, 1, 1)
            noisy_x = torch.sqrt(a) * x + torch.sqrt(1 - a) * e

            output = model(noisy_x, t, text_emb=text_emb)
            loss = criterion(output, e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update(model)
            # update_ema(ema, model, ema_decay)
            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f}")

    # Save checkpoint
    if rank == 0:
        os.makedirs("checkpoints", exist_ok=True)
    
        print("Gathering full model from DTensor shards...")
        model_state = gather_state_dict(model)  # Convert DTensor → full tensor
    
        ckpt = {
            "weights": model_state,
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
        }
    
        torch.save(ckpt, "checkpoints/ddpm_text2img_checkpoint.pt")
        print("✅ Checkpoint saved as full tensor at checkpoints/ddpm_text2img_checkpoint.pt")
    
    dist.barrier()

def main():
    dataset = load_dataset("jackyhate/text-to-image-2M", split="train", streaming=True)

    # Load CLIP tokenizer & text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()

    # Start training
    train(
        dataset=dataset,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        num_epochs=2,
        batch_size=4,
        lr=2e-5
    )

if __name__ == '__main__':
    main()
