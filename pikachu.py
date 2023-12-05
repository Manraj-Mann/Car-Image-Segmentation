from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
from keras import backend
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class Model:

    model = None
    def __init__(self) -> None:
        self.load_model()
    
    def load_model(self):
        
        # Load the model from the saved file
        loaded_autoencoder = load_model("autoencoder_model.h5")
        self.model = loaded_autoencoder

    def process_custom_image(self , image):

        IMAGE_NON_SEEN = image
        RESIZED_NON_SEEN = cv2.resize(IMAGE_NON_SEEN,(300,300))
        RESIZED_NON_SEEN = RESIZED_NON_SEEN.reshape(1,RESIZED_NON_SEEN.shape[0],RESIZED_NON_SEEN.shape[1],RESIZED_NON_SEEN.shape[2])

        Prediction_Non_Seen = self.model.predict(RESIZED_NON_SEEN)

        print(type(Prediction_Non_Seen[0]))
        bgr_image = cv2.cvtColor(Prediction_Non_Seen[0], cv2.COLOR_RGB2BGR)

        return Prediction_Non_Seen[0]
        # cv2.imshow('Image', bgr_image)
        # cv2.waitKey(0)

        # figure,axis = plt.subplots(1,1,figsize=(15,15))
        # axis.imshow(Prediction_Non_Seen[0])
        # axis.axis('off')
        # plt.tight_layout()
        # plt.savefig("result.jpg")
        # image = cv2.imread("result.jpg")

        # return image    

# if __name__ == "__main__":

#     print("Loading Model ....")
#     model = Model()
#     model.load_model()
#     print(model.model)
#     print("Loaded Model ....")
#     image = model.process_custom_image("OIPq.jpg")
#     cv2.imshow("image" , image)
#     cv2.waitKey(0)
    
