import torch
import random
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint import state_dict as dist_state_dict
from torch.distributed.checkpoint import load_state_dict as dist_load_state_dict

def cast_tuple(val, length: int = None) -> tuple:
    '''
    Casts input to a tuple. If the input is a list, converts it to a tuple. If input a single value, casts it to a
        tuple of length `length`, which is 1 if not provided.
    '''
    if isinstance(val, list):
        val = tuple(val)
    if length==None:
        length=1
    output = val if isinstance(val, tuple) else ((val,) * length)

    if length!=None:
        assert len(output) == length

    return output

def gather_state_dict(model):
    """
    The training occurs using FSDP, this function gathers the distributed state_dict and combines it to a single dict.
    """
    try:
        full_state = {}
        dist_state = model.state_dict()
        # Materialize shards into full tensors
        for k, v in dist_state.items():
            if hasattr(v, "to_local"):  # If DTensor
                full_state[k] = v.full_tensor()
            else:
                full_state[k] = v
        return full_state
    except Exception as e:
        print(f"[Warning] Could not gather DTensor state_dict directly: {e}")
        # Fallback: move everything to CPU if not DTensor
        return {k: v.to("cpu") for k, v in model.state_dict().items()}


# --------------------------
# Dataset
# --------------------------
# TO-DO:
# Right now the model is iterable and hence can't be shulffed, this might cause training bias, need to make it shuffable.
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, rank, world_size, tokenizer, image_size=256):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                img = item["jpg"]  # PIL.Image
                caption = item["json"]["prompt"]  # str
                img_tensor = self.transform(img)  # Convert to tensor
                yield img_tensor, caption

# --------------------------
# Seed
# --------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------
# Custom EMA
# --------------------------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        # Initialize shadow parameters on the same device as model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().to(param.device)
                self.original[name] = param

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue

            ema_param = self.shadow[name]

            # Ensure both tensors are on the same device
            if ema_param.device != param.device:
                ema_param = ema_param.to(param.device)
                self.shadow[name] = ema_param  # update shadow reference

            # Perform EMA update
            ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
