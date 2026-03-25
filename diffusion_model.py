import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import pandas as pd
import einops

from diffusers import StableDiffusionPipeline, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel

class sketch_dataset(Dataset):
    def __init__(self, csv_path, image_root):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root

        self.img_tf = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def edge_sketch(self, img):
        gray = img.mean(dim=0, keepdim=True)             
        edges = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
        edges = F.pad(edges, (0, 1, 0, 0))
        return edges

    def __len__(self):
        return len(self.data)

   def __getitem__(self, idx):
    row = self.data.iloc[idx]
    img_path = os.path.join(self.image_root, row["image"])
    caption = row["caption"]

    if not isinstance(caption, str) or len(caption.strip()) < 5:
        return None

    if len(caption.split()) > 50:
        caption = " ".join(caption.split()[:50])

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        print(f"Skipping corrupted image: {img_path}")
        return None

    img = self.img_tf(img)

    if torch.isnan(img).any():
        return None

    if img.std() < 0.01:
        return None

    sketch = self.edge_sketch(img)
    sketch = sketch.repeat(3, 1, 1)

        return {
            "image": img,
            "sketch": sketch,
            "text": caption
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return {
        "image": torch.stack([b["image"] for b in batch]),
        "sketch": torch.stack([b["sketch"] for b in batch]),
        "text": [b["text"] for b in batch]
    }

class RegionPooler(nn.Module):
    def __init__(self, in_channels=512, region_dim=256):htr00055thy
        super().__init__()
        self.proj = nn.Conv2d(in_channels, region_dim, 4, 4)
        self.norm = nn.LayerNorm(region_dim)

    def forward(self, h):
        r = self.proj(h)
        r = einops.rearrange(r, "b d h w -> b (h w) d")
        r = self.norm(r)
        return r


class RegionTextAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, regions, text_tokens):
        out, weights = self.attn(
            query=regions,
            key=text_tokens,
            value=text_tokens
        )
        return out, weights

def main():
    CSV_PATH = "/workspace/captions.csv"
    IMAGE_ROOT = "/workspace/flickr30k-images"
    BATCH_SIZE = 4         
    EPOCHS = 5              
    LAMBDA = 0.1
    LR = 1e-4
    SAVE_PATH = "controlnet_dual_stage.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset = sketch_dataset(CSV_PATH, IMAGE_ROOT)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    diff_model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    diff_model.enable_xformers_memory_efficient_attention()

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype=torch.float16
    ).to(device)

    diff_model.controlnet = controlnet
    scheduler = diff_model.scheduler

    diff_model.unet.requires_grad_(False)
    diff_model.vae.requires_grad_(False)
    diff_model.text_encoder.requires_grad_(False)

   
    region_pooler = RegionPooler().to(device)
    region_attn = RegionTextAttention().to(device)
    text_proj = nn.Linear(768, 256).to(device)

    optimizer = torch.optim.AdamW(
        list(controlnet.parameters()) +
        list(region_pooler.parameters()) +
        list(region_attn.parameters()) +
        list(text_proj.parameters()),
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler()

    tokenizer = diff_model.tokenizer
    text_encoder = diff_model.text_encoder
    vae = diff_model.vae
    unet = diff_model.unet
    for epoch in range(EPOCHS):

        epoch_diff_loss = 0
        epoch_sem_loss = 0

        for step, batch in enumerate(dataloader):

            if batch is None:
                continue

            images = batch["image"].to(device, dtype=torch.float16, non_blocking=True)
            sketches = batch["sketch"].to(device, dtype=torch.float16, non_blocking=True)
            texts = batch["text"]

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)
            t = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            )

            noisy_latents = scheduler.add_noise(latents, noise, t)

            with torch.no_grad():
                tokens = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(device)

                text_embeds = text_encoder(tokens)[0]

            with torch.cuda.amp.autocast():

                down_samples, mid_sample = controlnet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeds,
                    controlnet_cond=sketches,
                    return_dict=False
                )

                noise_pred = unet(
                    noisy_latents,
                    t,
                    encoder_hidden_states=text_embeds,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample

                diffusion_loss = F.mse_loss(noise_pred, noise)

                projected_text = text_proj(text_embeds)
                regions = region_pooler(down_samples[-1].float())
                region_out, _ = region_attn(regions, projected_text)

                sim = torch.einsum("brd,btd->brt", region_out, projected_text)
                semantic_loss = -F.normalize(sim, dim=-1).max(dim=-1)[0].mean()

                total_loss = diffusion_loss + LAMBDA * semantic_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_diff_loss += diffusion_loss.item()
            epoch_sem_loss += semantic_loss.item()

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Avg Diffusion Loss: {epoch_diff_loss/len(dataloader):.4f}")
        print(f"Avg Semantic Loss: {epoch_sem_loss/len(dataloader):.4f}")

        torch.save({
            "controlnet": controlnet.state_dict(),
            "region_pooler": region_pooler.state_dict(),
            "region_attn": region_attn.state_dict(),
            "text_proj": text_proj.state_dict()
        }, f"checkpoint_epoch_{epoch+1}.pt")

    print("Training complete.")

if __name__ == "__main__":
    main()
