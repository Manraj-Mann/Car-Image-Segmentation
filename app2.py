import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mode import MyModal as model  # Assuming Model is the correct class name

st.title("AI Image Cropping")
st.caption("AI Image cropping using U-Net Segementation")

if 'messages' not in st.session_state:
    # If not, initialize it as an empty list
    st.session_state.messages = []

for message in st.session_state.messages:
    st.image(message['image'] , use_column_width=True)

class UI:
    
    def __init__(self):
        self.model = Model()

    def displayUI(self):
        uploaded_images = st.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if uploaded_images:
            for image in uploaded_images:
                input_image = Image.open(image)  # read image
                st.image(input_image, caption='Original Image', use_column_width=True)

                # if st.button("Segment Image"):
                with st.spinner("🤖 AI is at Work! "):

                    generated_images = self.model.get_response(input_image)
                
                    st.success("Successfully Generated!")

                    concatenated_image = Image.new('RGB', (256 + 256 + 256, 256))
                    concatenated_image.paste(generated_images[0], (0, 0))
                    concatenated_image.paste(generated_images[1], (256, 0))
                    concatenated_image.paste(generated_images[2], (512 , 0))
                    st.session_state.messages.append({'image': concatenated_image})
                    
                    st.image(concatenated_image , caption="Generated Image" , use_column_width=True)

                    st.balloons()

        else:
            st.write("Upload car an Image here !")

class Model:
    def __init__(self):
        self.model_m = model()

    def get_response(self, image):
        return self.model_m.process_custom_image(image)

def main():
    ui = UI()
    ui.displayUI()

if __name__ == "__main__":
    main()
