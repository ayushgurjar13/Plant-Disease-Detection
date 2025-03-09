import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import streamlit as st
from PIL import Image

# Add CSS for background
def set_background_image(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url({image_path});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load your model
model = load_model('plant_disease_prediction_model.h5')

# Define the 38 plant disease classes and their descriptions
labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Disease descriptions (add all descriptions here)
disease_info = {
    "Apple___Apple_scab": "Apple scab is a common fungal disease that affects apple trees, causing dark, scabby lesions on the leaves, fruit, and stems. It can lead to significant crop loss if not controlled. Management includes pruning, using resistant varieties, and applying fungicides.",
    "Apple___Black_rot": "Black rot is a fungal disease of apple trees that results in fruit rot, leaf spots, and cankers on branches. It is especially damaging in humid conditions. Control involves removing infected material and applying fungicides.",
    "Apple___Cedar_apple_rust": "Cedar apple rust is a fungal disease affecting apples and junipers, characterized by yellow-orange spots on apple leaves and galls on junipers. Management includes removing nearby junipers and using resistant apple varieties.",
    "Apple___healthy": "This apple tree is healthy with no visible signs of disease or pest infestation. Regular monitoring and proper care are essential to maintain tree health.",
    "Cherry_(including_sour)___Powdery_mildew": "Powdery mildew is a fungal disease of cherry trees that appears as a white, powdery coating on leaves, buds, and stems. It thrives in warm, dry conditions. Management includes proper spacing, pruning, and fungicide applications.",
    "Cherry_(including_sour)___healthy": "This cherry tree is healthy with no visible signs of disease or pest infestation. Proper care, such as pruning and regular monitoring, is crucial to maintaining tree health.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray leaf spot is a fungal disease in corn caused by *Cercospora zeae-maydis*, resulting in rectangular, tan-colored lesions on leaves. It reduces photosynthesis, impacting yield. Fungicides and resistant hybrids help manage the disease.",
    "Corn_(maize)___Common_rust_": "Common rust is a fungal disease in corn characterized by reddish-brown pustules on leaves. It can lead to reduced photosynthesis and lower yields. Resistant varieties and fungicides are effective controls.",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern leaf blight is a fungal disease that causes cigar-shaped lesions on corn leaves, leading to reduced photosynthesis and yield loss. Management includes crop rotation, resistant hybrids, and fungicide applications.",
    "Corn_(maize)___healthy": "This corn plant is healthy with no visible signs of disease or pest infestation. Proper care, including balanced fertilization and pest management, ensures good growth.",
    "Grape___Black_rot": "Black rot is a fungal disease in grapevines, causing dark, sunken spots on leaves and rotting berries. It thrives in warm, wet conditions. Effective management includes pruning, removing infected parts, and applying fungicides.",
    "Grape___Esca_(Black_Measles)": "Esca, also known as black measles, is a complex disease in grapevines causing leaf discoloration and vine decline. It is caused by multiple fungi. Management includes pruning and removing infected parts.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf blight in grapes, caused by *Isariopsis clavispora*, results in brown spots and defoliation, reducing the plant's ability to photosynthesize. Control includes improving air circulation and applying fungicides.",
    "Grape___healthy": "This grapevine is healthy with no visible signs of disease or pest infestation. Proper pruning and monitoring are essential to maintain health.",
    "Orange___Haunglongbing_(Citrus_greening)": "Huanglongbing, or citrus greening, is a bacterial disease spread by psyllids. It causes yellowing of leaves, fruit deformation, and tree decline. Control includes removing infected trees and controlling psyllid populations.",
    "Peach___Bacterial_spot": "Bacterial spot in peach trees causes dark, sunken spots on fruit and leaves, leading to defoliation and fruit loss. Management includes using resistant varieties and applying bactericides.",
    "Peach___healthy": "This peach tree is healthy with no visible signs of disease or pest infestation. Regular pruning, proper irrigation, and pest management ensure good growth.",
    "Pepper,_bell___Bacterial_spot": "Bacterial spot in bell peppers causes water-soaked lesions on leaves and fruits, leading to defoliation and fruit loss. Management includes using resistant varieties and applying bactericides.",
    "Pepper,_bell___healthy": "This bell pepper plant is healthy with no visible signs of disease or pest infestation. Proper care, including balanced fertilization and pest monitoring, ensures good growth.",
    "Potato___Early_blight": "Early blight is a fungal disease in potatoes caused by *Alternaria solani*, resulting in brown spots with concentric rings on leaves. It can lead to reduced yield. Management includes crop rotation, resistant varieties, and fungicides.",
    "Potato___Late_blight": "Late blight is a devastating fungal disease caused by *Phytophthora infestans*. It leads to brown lesions on leaves and rotting tubers. Fungicides and resistant varieties are key controls.",
    "Potato___healthy": "This potato plant is healthy with no visible signs of disease or pest infestation. Proper watering and pest control are essential for maintaining health.",
    "Strawberry___Leaf_scorch": "Leaf scorch in strawberries, caused by *Diplocarpon earliana*, results in dark spots on leaves and defoliation. Control includes proper irrigation, spacing, and fungicide applications.",
    "Strawberry___healthy": "This strawberry plant is healthy with no visible signs of disease or pest infestation. Proper fertilization and pest monitoring ensure good growth.",
    "Tomato___Bacterial_spot": "Bacterial spot in tomatoes causes water-soaked lesions on leaves, stems, and fruits, leading to defoliation and fruit loss. Control includes crop rotation, resistant varieties, and bactericides.",
    "Tomato___Early_blight": "Early blight is a fungal disease in tomatoes caused by *Alternaria solani*. It causes dark spots on leaves with concentric rings, reducing yield. Management includes resistant varieties and fungicides.",
    "Tomato___Late_blight": "Late blight is a fungal disease in tomatoes caused by *Phytophthora infestans*, leading to brown lesions on leaves and fruit rot. Control includes resistant varieties and fungicides.",
    "Tomato___Leaf_Mold": "Leaf mold in tomatoes is caused by *Passalora fulva*. It creates yellow spots on the top of leaves and mold on the underside. Proper ventilation and fungicides can control the disease.",
    "Tomato___Septoria_leaf_spot": "Septoria leaf spot in tomatoes causes small, circular lesions on leaves, leading to defoliation and yield loss. Management includes crop rotation, fungicides, and proper spacing.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider mites cause stippling and webbing on tomato leaves, leading to defoliation. Control includes predatory insects, proper watering, and miticides.",
    "Tomato___Target_Spot": "Target spot in tomatoes is caused by *Corynespora cassiicola*. It results in circular lesions with concentric rings on leaves and fruit. Management includes fungicides and resistant varieties.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "This virus in tomatoes causes yellowing and curling of leaves, stunted growth, and poor yield. Control includes using resistant varieties and managing whitefly populations.",
    "Tomato___Tomato_mosaic_virus": "Tomato mosaic virus causes mottling, yellowing, and stunted growth in tomato plants. Management includes crop rotation, resistant varieties, and hygiene practices.",
    "Tomato___healthy": "This tomato plant is healthy, with no visible signs of disease or pest infestation. Regular care, including proper watering, fertilization, and pest management, is essential for maintaining plant health."
}

# Prediction function
def predict_disease(image_path):
    img = load_img(image_path, target_size=(225, 225))  # Adjust image size
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x.astype('float32') / 255.0  # Normalize image
    predictions = model.predict(x)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    description = disease_info.get(predicted_label, "No additional information available.")
    return predicted_label, description

# Set background image
set_background_image("static/images/background.jpg")  # Replace with your background image path

# Streamlit app layout
st.title("🌱 Plant Disease Detection")
st.markdown("Upload a plant leaf image to identify the disease and get details about it.")

# File upload
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the file locally
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Predict disease
    with st.spinner('Analyzing the image...'):
        disease, description = predict_disease(temp_path)

    # Display results
    st.success(f"**Predicted Disease:** {disease}")
    st.info(f"**Description:** {description}")

    # Cleanup temp file
    os.remove(temp_path)
