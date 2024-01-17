from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from langserve import add_routes
from my_app.chain import chain as my_app_chain
from diffusers import DiffusionPipeline, AutoencoderKL
from PIL import Image
import torch
import numpy as np
from io import BytesIO
import uvicorn

app = FastAPI()

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
refiner.to("cuda")

def generate_image(prompt: str):
    n_steps = 40
    high_noise_frac = 0.7

    image = pipe(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images
    image = refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]

    image_np = np.array(image)
    return image_np

@app.get("/generate/{prompt}")
def generate(prompt: str):
    image_np = generate_image(prompt)

    pil_image = Image.fromarray(image_np)

    image_buffer = BytesIO()
    pil_image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    return StreamingResponse(image_buffer, media_type="image/png")

add_routes(app, my_app_chain, path="/my-app")
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
