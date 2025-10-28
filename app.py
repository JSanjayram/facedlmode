import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", page_icon="🎭", layout="wide")

@st.cache_resource
def load_face_model():
    try:
        model = load_model('accurate_mask_detector.h5')
        return model
    except:
        st.error("Model file not found. Please ensure 'accurate_mask_detector.h5' exists.")
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
    st.title("🎭 Face Mask Detection System")
    st.markdown("**AI-powered mask detection with your trained model**")
    
    model = load_face_model()
    face_cascade = load_face_cascade()
    
    if model is None:
        return
    
    # Sidebar
    st.sidebar.header("Detection Options")
    mode = st.sidebar.selectbox("Choose Mode", ["Image Upload", "Video Upload", "Batch Processing"])
    
    if mode == "Image Upload":
        st.header("📸 Single Image Detection")
        
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            processed_image = process_frame(image_np, model, face_cascade)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Detection Result")
                st.image(processed_image, use_column_width=True)
    
    elif mode == "Video Upload":
        st.header("🎬 Video Processing")
        
        uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 5 == 0:  # Process every 5th frame for speed
                    processed_frame = process_frame(frame, model, face_cascade)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(processed_frame, channels="RGB", use_column_width=True)
                
                frame_count += 1
            
            cap.release()
            os.unlink(tfile.name)
    
    elif mode == "Batch Processing":
        st.header("📁 Batch Image Processing")
        
        uploaded_files = st.file_uploader("Upload multiple images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Image {i+1}: {uploaded_file.name}")
                
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                if len(image_np.shape) == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                processed_image = process_frame(image_np, model, face_cascade)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, use_column_width=True, caption="Original")
                with col2:
                    st.image(processed_image, use_column_width=True, caption="Detection")
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.info("""
    **Architecture**: Custom CNN
    **Input Size**: 128x128 RGB
    **Classes**: Mask / No Mask
    **Accuracy**: 95%+
    **Framework**: TensorFlow/Keras
    """)

if __name__ == "__main__":
    main()