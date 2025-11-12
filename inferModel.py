import torch
from huggingface_hub import hf_download_hub
from modelClasses import UNET
import torchvision.transforms as T
import matplotlib.pyplot as plt

def sample(model, text_emb, img_size=(1, 3, 64, 64)):
    model.eval()
    x = torch.randn(img_size).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        with torch.no_grad():
            eps = model(x, t_tensor, text_emb=text_emb)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps) + torch.sqrt(betas[t]) * noise
    return x


if __name__ = "__main__":
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initalize the model
    model = UNET(text_dim=512).to(device)

    # download model checkpoint from huggingface
    ckpt_path = hf_download_hub(
            repo_id="sharmaarush/text-2-img",
            filename="text2img_step_6000.pt"
        )
    
    # load the model checkpoints for eval
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['weights'], strict=False) # resolves layer unmatch error
    model.eval()

    # dummy data (need to handle user inputs validation)
    B, C, H, W = 1, 3, 64, 64
    x_t = torch.randn(B, C, H, W).to(device)  # random noise input
    t = torch.tensor([500], dtype=torch.long).to(device)  # timestep

    # Example: text embedding (dummy)
    text_emb = torch.randn(B, 77, 512).to(device)
    with torch.no_grad():
        noise_pred = model(x_t, t, text_emb=text_emb)
    
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    gen_img = sample(model, text_emb)

    # visualize the output
    to_pil = T.ToPILImage()
    img = gen_img[0].cpu().clamp(0, 1)
    plt.imshow(to_pil(img))
    plt.axis("off")
    plt.show()
