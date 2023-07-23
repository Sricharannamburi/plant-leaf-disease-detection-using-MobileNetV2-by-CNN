import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

# Load the trained model
model = load_model(r'C:\Users\sricharan namburi\Downloads\Plant_Disease_ML_Model2-main\Plant_Disease_ML_Model2-main\models\pddmobilenet2v1.h5')
'''{'Apple___Apple_scab': 0,
 'Apple___Black_rot': 1,
 'Apple___Cedar_apple_rust': 2,
 'Apple___healthy': 3,
 'Grape___Black_rot': 4,
 'Grape___Esca_(Black_Measles)': 5,
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 6,
 'Grape___healthy': 7,
 'Potato___Early_blight': 8,
 'Potato___healthy': 9,
 'Tomato___Early_blight': 10,
 'Tomato___Leaf_Mold': 11,
 'Tomato___Septoria_leaf_spot': 12,
 'Tomato___Spider_mites Two-spotted_spider_mite': 13,
 'Tomato___Target_Spot': 14,'''
# Set path to the folder containing images
path = r"C:\Users\sricharan namburi\OneDrive\Desktop\DATA\test1"

# Loop through all files in the folder
for filename in os.listdir(path):
    print(f'Processing {filename}...')
    # Check if file is an image
    if filename.endswith('.JPG') or filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.jfif') or filename.endswith('.png') or filename.endswith('.PNG'):
        # Load image and preprocess it
        
        img = image.load_img(os.path.join(path, filename), target_size=(256, 256))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Make prediction on the image
        pred = model.predict(img)
        
        # Get the label with highest probability
        label = np.argmax(pred)
        
        # Print the label and filename
        print(f'{filename}: {label}')
        
        # Display the image
        img = cv2.imread(os.path.join(path, filename))
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        
# Close all windows
cv2.destroyAllWindows()
