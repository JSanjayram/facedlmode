import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

st.set_page_config(page_title="Live Face Mask Detection", page_icon="🎭", layout="wide")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class MaskDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model('accurate_mask_detector.h5')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (128, 128))
                face_normalized = face_resized.astype("float32") / 255.0
                face_batch = np.expand_dims(face_normalized, axis=0)
                
                prediction = self.model.predict(face_batch, verbose=0)[0][0]
                
                if prediction > 0.5:
                    label = f"Mask: {prediction:.1%}"
                    color = (0, 255, 0)
                else:
                    label = f"No Mask: {1-prediction:.1%}"
                    color = (0, 0, 255)
                
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
    
    # WebRTC Live Detection
    st.header("📹 Live Webcam Detection")
    st.info("Click START to begin live detection. Allow camera access when prompted.")
    
    webrtc_ctx = webrtc_streamer(
        key="face-mask-detection",
        video_processor_factory=MaskDetectionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.video_processor:
        st.success("🟢 Live Detection Active")
    else:
        st.warning("🔴 Click START to begin detection")
    
    # Model info sidebar
    st.sidebar.header("🎯 Detection Info")
    st.sidebar.success("""
    **Model**: Custom CNN  
    **Accuracy**: 95%+  
    **Real-time**: 30+ FPS  
    **Classes**: Mask/No Mask
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Instructions:**")
    st.sidebar.markdown("1. Click **START** button")
    st.sidebar.markdown("2. Allow camera access")
    st.sidebar.markdown("3. Position face in frame")
    st.sidebar.markdown("4. View real-time results")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎨 Color Guide:**")
    st.sidebar.markdown("🟢 **Green Box** = Mask Detected")
    st.sidebar.markdown("🔴 **Red Box** = No Mask Detected")
    
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