import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model
model = load_model('model_optimal.h5')


# Define the label dictionary
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def predict_emotion(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Check if the image is in the correct shape
    if img_array.ndim == 2:  # if it's grayscale
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    result = model.predict(img_array)
    result = list(result[0])
    
    # Get the index of the maximum value
    img_index = result.index(max(result))
    
    # Print the predicted emotion
    predicted_emotion = label_dict[img_index]
    print(f"Predicted Emotion: {predicted_emotion}")
    
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted Emotion: {predicted_emotion}')
    plt.show()

# Example usage:
# Replace 'path_to_your_image.jpg' with the path to the image you want to analyze
predict_emotion('im37.png')
