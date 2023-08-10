import streamlit as st
from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

# Create the Streamlit app title and description
st.title("Stable Bud")
st.write("Generate Images using Stable Diffusion Pipeline")

# Create an input text area for user input
prompt = st.text_area("Enter a prompt:")

# Load the model and set up the pipeline
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Change this to "cuda" if GPU is available, otherwise "cpu"
# Replace these values with the actual model ID, revision, and auth token
revision = "fp16"


# Load the model using the correct attribute name ("from_pretrained")
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision=revision, dtype=torch.float16, use_auth_token=auth_token)

pipe.to(device)

# Create a button to generate the image
if st.button("Generate"):
    with autocast(device):
        image = pipe(prompt, guidance_scale=8.5)["sample"][0]

    # Save the generated image
    image.save('generatedimage.png')

    # Display the generated image
    st.image(Image.open('generatedimage.png'), caption='Generated Image')

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Stable Bud App", page_icon=":paintbrush:")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('deprecation.showPyplotGlobalUse', False)
