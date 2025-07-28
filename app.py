import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="ASL to Text Translator",
    page_icon="🤟",
    layout="wide"
)

# Title and description
st.title("🤟 ASL to Text Translator")
st.markdown("A real-time American Sign Language to Text translator using deep learning and computer vision.")

# Sidebar for ASL chart
with st.sidebar:
    st.header("📚 ASL Reference Chart")
    if os.path.exists("asl_chart.png"):
        st.image("asl_chart.png", caption="ASL Alphabet Chart")
    else:
        st.info("ASL chart image not found. Please add 'asl_chart.png' to the project directory.")

# Load model
@st.cache_resource
def load_asl_model():
    model_path = os.path.join('models', 'asl_cnn.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("Model not found! Please train the model first.")
        return None

model = load_asl_model()

# Class labels
@st.cache_data
def get_class_labels():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory('dataset', target_size=(64, 64), batch_size=1, class_mode='categorical')
    class_labels = [None] * len(gen.class_indices)
    for label, idx in gen.class_indices.items():
        class_labels[idx] = label
    return class_labels

class_labels = get_class_labels()

# Prediction function
def predict_asl_gesture(image):
    if model is None:
        return "Model not loaded"
    
    # Preprocess image
    img = cv2.resize(image, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)
    confidence = preds[0][idx]
    
    if class_labels and idx < len(class_labels):
        label = class_labels[idx]
        # Map special classes
        if label.lower() == 'space':
            return ' ', confidence
        if label.lower() == 'delete':
            return '<', confidence
        if label.lower() == 'nothing':
            return 'Nothing', confidence
        return label, confidence
    return str(idx), confidence

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📹 Webcam Input")
    
    # Camera input
    camera_input = st.camera_input("Take a picture of your hand sign")
    
    if camera_input is not None:
        # Convert to numpy array
        image = Image.open(camera_input)
        image_np = np.array(image)
        
        # Display the captured image
        st.image(image_np, caption="Captured Image", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Predict ASL Sign", type="primary"):
            with st.spinner("Analyzing hand sign..."):
                prediction, confidence = predict_asl_gesture(image_np)
                
                # Display result
                st.success(f"**Predicted:** {prediction}")
                st.info(f"**Confidence:** {confidence:.2%}")
                
                # Add to session state for text output
                if 'text_output' not in st.session_state:
                    st.session_state.text_output = ""
                
                if prediction not in ['Nothing', 'Model not loaded']:
                    if prediction == '<':
                        # Delete last character
                        st.session_state.text_output = st.session_state.text_output[:-1] if st.session_state.text_output else ""
                    elif prediction == ' ':
                        st.session_state.text_output += " "
                    else:
                        st.session_state.text_output += prediction

with col2:
    st.header("📝 Output Text")
    
    # Initialize text output
    if 'text_output' not in st.session_state:
        st.session_state.text_output = ""
    
    # Display current text
    st.text_area("Translated Text:", value=st.session_state.text_output, height=200, key="output_text")
    
    # Clear button
    if st.button("🗑️ Clear Text"):
        st.session_state.text_output = ""
        st.rerun()
    
    # Download button
    if st.session_state.text_output:
        st.download_button(
            label="💾 Download Text",
            data=st.session_state.text_output,
            file_name="asl_translation.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("**Built with:** Streamlit, OpenCV, MediaPipe, TensorFlow, Keras")
st.markdown("**Dataset:** [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Position your hand** in the camera view
    2. **Show an ASL sign** (A-Z, Space, Delete, or Nothing)
    3. **Click 'Predict ASL Sign'** to analyze
    4. **View the result** and translated text
    5. **Use the sidebar** for ASL reference
    6. **Clear or download** your text as needed
    """) 