import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

CSV_PATH = "/workspace/captions.csv"
IMAGE_ROOT = "/workspace/flickr30k-images"
CHECKPOINT_PATH = "checkpoint_epoch_5.pt"
OUTPUT_PATH = "generated_output.png"

device = "cuda" if torch.cuda.is_available() else "cpu"


data = pd.read_csv(CSV_PATH)
row = data.iloc[0]
image_name = row["image"]
prompt = row["caption"]

image_path = os.path.join(IMAGE_ROOT, image_name)
original_image = Image.open(image_path).convert("RGB")


tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

img_tensor = tf(original_image)
gray = img_tensor.mean(dim=0, keepdim=True)
edges = torch.abs(gray[:, :, 1:] - gray[:, :, :-1])
edges = F.pad(edges, (0, 1, 0, 0))
sketch_tensor = edges.repeat(3, 1, 1)
sketch = transforms.ToPILImage()(sketch_tensor)


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
controlnet.load_state_dict(checkpoint["controlnet"])


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
except:
    print("xformers not available, continuing without it")


image = pipe(
    prompt=prompt,
    image=sketch,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save(OUTPUT_PATH)
print(f"Image generated and saved to {OUTPUT_PATH}")
print(f"Prompt used: {prompt}")
