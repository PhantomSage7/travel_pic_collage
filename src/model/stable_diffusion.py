from diffusers import StableDiffusionPipeline
import torch

pipe = None

def load_model():
    global pipe
    if pipe is None:
        model_id = "CompVis/stable-diffusion-v1-4"
        precision = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=precision,
            revision="fp16" if torch.cuda.is_available() else "main"
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            pipe.enable_xformers_memory_efficient_attention()
        pipe.scheduler.config.num_train_timesteps = min(pipe.scheduler.config.num_train_timesteps, 1000)
    return pipe

def generate_images(prompt, num_images, resolution):
    pipe = load_model()
    width, height = map(int, resolution.split('x'))
    images = pipe([prompt] * num_images, height=height, width=width).images
    return images
