from diffusers import StableDiffusionPipeline
import torch

# Global variable to hold the model
pipe = None

def load_model():
    global pipe
    if pipe is None:  # Load the model only if it hasn't been loaded yet
        model_id = "CompVis/stable-diffusion-v1-4"
        
        # Set precision based on availability of CUDA
        precision = torch.float16 if torch.cuda.is_available() else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=precision,  
            revision="fp16" if torch.cuda.is_available() else "main"  # Use fp16 for GPU, otherwise default
        )
        # Load onto CUDA if available, else use CPU
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # Enable xformers only if CUDA is available
        if torch.cuda.is_available():
            pipe.enable_xformers_memory_efficient_attention()

    return pipe

def generate_images(prompt, num_images, resolution):
    # Ensure the model is loaded
    pipe = load_model()

    # Parse resolution
    width, height = map(int, resolution.split('x'))

    # Generate all images at once if possible
    images = pipe([prompt] * num_images, height=height, width=width).images
    
    return images
