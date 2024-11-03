
# Travel Photo Collage Generator

This project uses a **Stable Diffusion** model to generate travel-themed photo collages based on user-provided text prompts. It provides an interactive web interface, allowing users to input a theme or specific details (like "beaches in Bali" or "European castles") and receive a photo collage in response. The application is deployed using **Streamlit** and **Docker**, making it easy to set up and run on different platforms.

## Features
- **Text-to-Image Generation**: Leverages Stable Diffusion to interpret text prompts and generate corresponding travel-themed images.
- **Collage Creation**: Automatically arranges generated images into a cohesive collage for a visually pleasing result.
- **Streamlit Interface**: Simple and intuitive web interface for users to input prompts and view collages.
- **Containerized Deployment**: Deployed with Docker for easy setup and scaling.

## Business Use Case
The **Travel Photo Collage Generator** is designed for travel enthusiasts, influencers, or digital marketers looking to create visually engaging travel collages. This tool can be used to:
- **Enhance Social Media Content**: Travel influencers can quickly generate themed photo collages for platforms like Instagram and Pinterest.
- **Plan and Visualize Travel Itineraries**: Users can visualize their future travel destinations or create memory collages from past trips.
- **Marketing and Promotion**: Travel agencies and content creators can generate themed visual content for promotional materials or social media campaigns, attracting potential customers by showcasing exotic destinations.

---

## Project Structure

The repository is organized as follows:

```
📁 deployment
 ┣ 📄 Dockerfile         # Configuration file for Docker deployment
 ┣ 📄 Procfile           # Configuration for Heroku or similar platforms
📁 src
 ┣ 📁 model
 ┃ ┣ 📄 stable_diffusion.py  # Model loading and image generation
 ┣ 📁 utils
 ┃ ┣ 📄 image_processing.py  # Helper functions for image manipulation and collage creation
📁 static
 ┣ 📄 styles.css         # Styling for the Streamlit interface
📁 templates
 ┣ 📄 base.html          # HTML structure for Streamlit layout
📄 app.py                # Main application file for running Streamlit
📄 requirements.txt      # Dependencies for the application
📄 README.md             # Project documentation
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PhantomSage7/travel_pic_collage.git
   cd travel-pic-collage
   ```

2. **Install Dependencies**:
   Install required packages listed in `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application Locally**:
   Start the Streamlit application.
   ```bash
   streamlit run app.py
   ```

## Docker Deployment

1. **Build the Docker Image**:
   ```bash
   docker build -t travel-pic-collage -f deployment/Dockerfile .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 travel-pic-collage
   ```

## Cloud Deployment (Optional)
For cloud deployment, such as Heroku:
1. **Configure Environment Variables**: Set any required API keys or model configurations.
2. **Deploy Using Procfile**: Use `Procfile` configuration for compatibility with Heroku or similar PaaS.

## Technical Overview

### Model Selection
- **Stable Diffusion** was chosen for its ability to generate high-quality images based on text prompts. This model provides a balance between performance and visual appeal, making it ideal for creating travel-themed images that match user inputs.

### Image Generation Workflow
1. **Text Input**: The user enters a text prompt related to the desired travel theme.
2. **Model Processing**: Stable Diffusion interprets the prompt and generates corresponding images.
3. **Collage Assembly**: The images are processed and arranged in a collage layout using functions in `image_processing.py`.
4. **Display**: The collage is displayed in the Streamlit interface, ready for download or sharing.

## Challenges and Future Improvements
- **Image Quality Tuning**: Achieving consistent, travel-specific imagery required fine-tuning the model. Future improvements could involve fine-tuning Stable Diffusion on travel-specific datasets.
- **Layout Optimization**: Enhancing collage layout algorithms to better align images and prevent overlap.
- **Support for More Input Types**: Expanding to accept multiple prompts or themes in a single session, or supporting additional media like videos or animations.

## License
MIT License.
