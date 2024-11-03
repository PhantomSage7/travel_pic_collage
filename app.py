# app.py
import streamlit as st
from src.model.stable_diffusion import generate_images
from src.utils.image_processing import create_collage

st.title("Travel Photo Collage Generator")

# User input: text prompt
prompt = st.text_input("Enter a prompt for your travel collage:")
num_images = st.slider("Number of images", min_value=1, max_value=10, value=8)

# User input: select resolution
resolution = st.selectbox("Select Image Resolution", options=["512x512", "640x640", "768x768"], index=0)

# Generate images
if st.button("Generate Collage"):
    if prompt:
        with st.spinner("Generating images..."):
            images = generate_images(prompt, num_images, resolution)
            collage = create_collage(images)
            st.image(collage, caption="Generated Travel Collage")
    else:
        st.warning("Please enter a prompt to generate images.")