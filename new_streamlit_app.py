import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import requests
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .success-result {
        background: #1e3a2e;
        border: 1px solid #28a745;
        color: #00ff00;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .warning-result {
        background: #3a1e1e;
        border: 1px solid #dc3545;
        color: #ff4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .detection-card {
        background: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the TensorFlow 2.20 compatible model"""
    import os
    try:
        model_path = 'tiny_mask_detector.h5'
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process uploaded image for mask detection"""
    img_array = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = img_array[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        prediction = model.predict(face_batch, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        mask_status = 'With Mask' if prediction > 0.5 else 'Without Mask'
        color = (0, 255, 0) if prediction > 0.5 else (255, 0, 0)
        
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
        label = f"{mask_status}: {confidence*100:.1f}%"
        cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        results.append({
            'status': mask_status,
            'confidence': confidence * 100,
            'bbox': (x, y, w, h)
        })
    
    return img_array, results

def main():
    st.title("Face Mask Detection System")
    st.markdown("<div style='text-align: center;'><strong>AI-powered mask detection using CNN</strong></div>", unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.error("🚨 Model not found!")
        st.stop()
    
    st.success("Model Online!")
    
    tab1, tab2, tab3 = st.tabs(["Image Detection", "Model Info", "Live Camera"])
    
    with tab1:
        st.header("Image Detection")
        
        option = st.radio("Input method:", ["Upload File", "Image URL"], horizontal=True)
        
        image = None
        
        if option == "Upload File":
            uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
        else:
            image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
            if image_url:
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        
        if image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                with st.spinner("Analyzing image..."):
                    processed_img, results = process_image(image, model)
                
                st.image(processed_img, use_container_width=True)
                
                if results:
                    for i, result in enumerate(results):
                        status = result['status']
                        confidence = result['confidence']
                        
                        if status == 'With Mask':
                            st.markdown(f"""
                            <div class="success-result">
                                <strong>Face {i+1}:</strong> {status} 
                                <span style="float: right;">{confidence:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-result">
                                <strong>Face {i+1}:</strong> {status} 
                                <span style="float: right;">{confidence:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No faces detected in the image.")
    
    with tab2:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.code("""
Lightweight CNN:
- Conv2D(16, 3x3, ReLU)
- MaxPooling2D(2x2)
- Conv2D(32, 3x3, ReLU)
- MaxPooling2D(2x2)
- Flatten
- Dense(64, ReLU)
- Dense(1, Sigmoid)
            """)
        
        with col2:
            st.subheader("Model Performance")
            st.metric("Input Size", "64x64x3")
            st.metric("Classes", "2 (With/Without Mask)")
            st.metric("Model Size", "1.6MB")
    
    with tab3:
        st.header("Live Camera Detection")
        
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8)
        
        if 'live_detection' not in st.session_state:
            st.session_state.live_detection = False
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔴 Start Live Detection"):
                st.session_state.live_detection = True
        
        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.live_detection = False
        
        st.markdown(f"**Status:** {'🟢 LIVE' if st.session_state.live_detection else '🔴 STOPPED'}")
        
        if st.session_state.live_detection:
            # Continuous detection with auto-refresh
            st.markdown("**📹 Live Detection Active - Camera will refresh automatically**")
            
            camera_image = st.camera_input("Take photo for detection", key=f"live_cam_{int(time.time())}")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                processed_img, results = process_image(image, model)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(processed_img, use_container_width=True)
                
                with col2:
                    if results:
                        for result in results:
                            status = result['status']
                            confidence = result['confidence']
                            
                            if status == 'With Mask':
                                st.success(f"✅ MASK DETECTED - {confidence:.1f}%")
                            else:
                                st.error(f"⚠️ NO MASK - {confidence:.1f}%")
                    else:
                        st.info("👤 No faces detected")
            
            # Auto-refresh every 3 seconds for continuous detection
            time.sleep(3)
            st.rerun()
        else:
            st.info("Click 'Start Live Detection' to begin real-time mask detection")

if __name__ == "__main__":
    main()