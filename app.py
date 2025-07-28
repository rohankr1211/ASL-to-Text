import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import random

# Page config
st.set_page_config(
    page_title="ASL to Text Translator",
    page_icon="🤟",
    layout="wide"
)

# Title and description
st.title("🤟 ASL to Text Translator")
st.markdown("A real-time American Sign Language to Text translator using computer vision.")

# Sidebar for ASL chart
with st.sidebar:
    st.header("📚 ASL Reference Chart")
    if os.path.exists("asl_chart.png"):
        st.image("asl_chart.png", caption="ASL Alphabet Chart")
    else:
        st.info("ASL chart image not found. Please add 'asl_chart.png' to the project directory.")

# Mock prediction function (no TensorFlow needed)
def predict_asl_gesture(image):
    # Simple mock prediction for demo
    mock_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ', '<', 'Nothing']
    prediction = random.choice(mock_labels)
    confidence = random.uniform(0.7, 0.95)
    return prediction, confidence

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
                
                if prediction not in ['Nothing']:
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
st.markdown("**Built with:** Streamlit, OpenCV, Pillow")
st.markdown("**Demo Mode:** This is a demonstration version with mock predictions.")

# Instructions
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Position your hand** in the camera view
    2. **Show an ASL sign** (A-Z, Space, Delete, or Nothing)
    3. **Click 'Predict ASL Sign'** to analyze
    4. **View the result** and translated text
    5. **Use the sidebar** for ASL reference
    6. **Clear or download** your text as needed
    
    **Note:** This is a demo version. For full functionality, train and add your model.
    """) 