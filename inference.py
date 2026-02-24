import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble",
    torch_dtype=torch.float16
).to(device)

controlnet.load_state_dict(torch.load("controlnet_finetuned.pt"))
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.enable_xformers_memory_efficient_attention()

sketch = Image.open("test_sketch.png").convert("RGB")
prompt = "A girl wearing a red dress standing in a green forest"
image = pipe(
    prompt=prompt,
    image=sketch,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
image.save("generated_output.png")
print("Image generated and saved.")
