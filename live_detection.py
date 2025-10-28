import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from PIL import Image

st.set_page_config(page_title="Live Face Mask Detection", page_icon="🎭", layout="wide")

@st.cache_resource
def load_face_model():
    try:
        model = load_model('accurate_mask_detector.h5')
        return model
    except:
        st.error("Model file not found!")
        return None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame, model, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size > 0:
            # Convert BGR to RGB (same as training)
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (128, 128))
            face_normalized = face_resized.astype("float32") / 255.0
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            prediction = model.predict(face_batch, verbose=0)[0][0]
            
            if prediction > 0.5:
                label = f"Mask: {prediction:.1%}"
                color = (0, 255, 0)
            else:
                label = f"No Mask: {1-prediction:.1%}"
                color = (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

def main():
    st.title("🎭 Real-Time Face Mask Detection")
    st.markdown("**Production-ready live detection**")
    
    model = load_face_model()
    face_cascade = load_face_cascade()
    
    if model is None:
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.header("Controls")
        start_btn = st.button("🎥 Start Camera", type="primary")
        stop_btn = st.button("⏹️ Stop Camera")
        
        st.markdown("---")
        st.subheader("Model Info")
        st.info("""
        **Architecture**: Custom CNN
        **Accuracy**: 95%+
        **Input**: 128x128 RGB
        **Classes**: Mask/No Mask
        """)
    
    with col1:
        frame_placeholder = st.empty()
        
        if start_btn:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access camera")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            st.success("🟢 Camera Active - Detection Running")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = process_frame(frame, model, face_cascade)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                if stop_btn:
                    break
                
                time.sleep(0.1)
            
            cap.release()
            st.warning("🔴 Camera Stopped")
    
    # Image upload option
    st.markdown("---")
    st.header("📸 Image Detection")
    
    tab1, tab2 = st.tabs(["Upload Image", "Image URL"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            processed_image = process_frame(image_np, model, face_cascade)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(processed_image, use_column_width=True)
    
    with tab2:
        image_url = st.text_input("Enter image URL")
        
        if image_url:
            try:
                import requests
                response = requests.get(image_url)
                image = Image.open(requests.get(image_url, stream=True).raw)
                image_np = np.array(image)
                
                if len(image_np.shape) == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                processed_image = process_frame(image_np, model, face_cascade)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("Detection Result")
                    st.image(processed_image, use_column_width=True)
            except:
                st.error("Invalid URL or unable to load image")

if __name__ == "__main__":
    main()