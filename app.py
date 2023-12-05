import streamlit as st
from PIL import Image
import numpy as np
import cv2
from pikachu import Model as model  # Assuming Model is the correct class name

st.title("UnderWater Image Segmentation")
st.caption("Underwater image segmentation using autoencoders")

if 'messages' not in st.session_state:
    # If not, initialize it as an empty list
    st.session_state.messages = []

for message in st.session_state.messages:
    st.image(message['image'] , use_column_width=True)

class UI:

    def resize_images_to_equal_dimensions(self , img1, img2):
        # Choose a common height for both images
        common_height = 300
        common_width = 300
    
        # Resize both images to the same height and width
        img1_resized = img1.resize((common_width, common_height))
        img2_resized = img2.resize((common_width, common_height))

        return img1_resized, img2_resized
    

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

                    input_image_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
                    # Ensure the data type is uint8
                    generated_image = self.model.get_response(input_image_cv2)
                    generated_image = (generated_image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)

                    # Convert the cv2 image (NumPy array) to a PIL image
                    pil_image = Image.fromarray(generated_image)
                    st.success("Successfully Generated!")

                    st.image(pil_image , caption="Generated Image" , use_column_width=True)

                    img1 = input_image
                    img2 = pil_image
                    
                    img1_resized, img2_resized = self.resize_images_to_equal_dimensions(img1, img2)

                    # Concatenate images horizontally
                    concatenated_image = Image.new('RGB', (300 + 300, 300))
                    concatenated_image.paste(img1_resized, (0, 0))
                    concatenated_image.paste(img2_resized, (img1_resized.width, 0))
                    st.session_state.messages.append({'image': concatenated_image})

                    st.balloons()

        else:
            st.write("Upload an Image")

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
