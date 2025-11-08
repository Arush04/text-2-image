import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torch.distributed import init_process_group
from torch.distributed.tensor import DeviceMesh
from torch.distributed.fsdp import fully_shard, StateDictType
# from torch.distributed.fsdp.api import checkpoint_module
from tqdm import tqdm
from PIL import Image
import random
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from modelClasses import UNET, DDPM_Scheduler

# Example dataset wrapper for your HF IterableDataset
class TextImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, image_size=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __iter__(self):
        # iterator yields CPU tensors and caption strings only
        for item in self.dataset:
            img = item["jpg"]                      # PIL.Image
            caption = item["json"]["prompt"]       # str
            img_tensor = self.transform(img)       # CPU tensor [C,H,W], float32
            yield img_tensor, caption

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# -------- Training Loop --------
def train(batch_size: int = 8,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr: float = 2e-5,
          checkpoint_path: str = None,
          dataset=None,
          tokenizer=None,
          text_encoder=None):

    # --------------------------
    # Setup and Assertions
    # --------------------------
    set_seed(random.randint(0, 2**32 - 1)) if seed == -1 else set_seed(seed)
    assert dataset is not None, "Please pass a dataset"
    assert tokenizer is not None, "Please pass a tokenizer"
    assert text_encoder is not None, "Please pass a text encoder"

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())

    init_process_group(backend="nccl")

    # --------------------------
    # Dataset & Dataloader
    # --------------------------
    train_dataset = TextImageDataset(dataset, tokenizer, image_size=256)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    print(">>>>>>>>>> Data loader finished >>>>>>>>>>>>>")
    # --------------------------
    # Model and EMA setup
    # --------------------------
    from torch.distributed._tensor import init_device_mesh
    device_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("data_parallel",))
    print(f"Rank {rank} | Device Mesh: {device_mesh}")

    # Create model on meta device (saves memory)
    # with torch.device("meta"):
    model = UNET()

    # Create EMA before FSDP (must be a deep copy)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    # Now shard only the main model
    model = fully_shard(model, mesh=device_mesh)
    
    # Move both to GPU
    model = model.to_empty(device="cuda")
    ema.module = ema.module.to_empty(device="cuda")

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    scheduler.alpha = scheduler.alpha.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # --------------------------
    # Load checkpoint if available
    # --------------------------
    if checkpoint_path is not None and rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    dist.barrier()  # Sync all ranks after loading

    # --------------------------
    # Training Loop
    # --------------------------
    text_encoder = text_encoder.to("cuda")
    text_encoder.eval()
    print(">>>>>>>>>>> Starting training >>>>>>>>>>>")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img_batch, text_emb_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0)):
            x = img_batch.cuda(non_blocking=True)
            # Tokenize on CPU (fast) and move tokens to CUDA before encoding
            tokens = tokenizer(text_emb_batch,
                               padding='max_length',
                               truncation=True,
                               max_length=77,
                               return_tensors='pt')
            tokens = {k: v.to("cuda", non_blocking=True) for k, v in tokens.items()}
        
            with torch.no_grad():
                text_emb = text_encoder(**tokens).last_hidden_state

            t = torch.randint(0, num_time_steps, (x.shape[0],), device="cuda")
            e = torch.randn_like(x)

            a = scheduler.alpha[t].view(x.shape[0], 1, 1, 1)
            noisy_x = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

            # Forward pass
            output = model(noisy_x, t, text_emb=text_emb)
            loss = criterion(output, e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(train_loader):.5f}")

    # --------------------------
    # Save checkpoint (only rank 0)
    # --------------------------
    if rank == 0:
        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }
        torch.save(checkpoint, 'checkpoints/ddpm_text2img_checkpoint.pt')
        print("Checkpoint saved at checkpoints/ddpm_text2img_checkpoint.pt")

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
