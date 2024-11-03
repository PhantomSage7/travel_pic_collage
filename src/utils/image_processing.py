# src/utils/image_processing.py
from PIL import Image

def create_collage(images):
    # Calculate the size of the collage
    num_images = len(images)
    width, height = images[0].size
    collage_width = width * 2
    collage_height = height * ((num_images + 1) // 2)
    
    collage = Image.new("RGB", (collage_width, collage_height))
    
    for idx, image in enumerate(images):
        x = (idx % 2) * width
        y = (idx // 2) * height
        collage.paste(image, (x, y))
    
    return collage
