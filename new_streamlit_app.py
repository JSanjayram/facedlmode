import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import requests

# Configure Streamlit page
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force dark theme
st.markdown("""
<script>
const stApp = window.parent.document.querySelector('.stApp');
if (stApp) {
    stApp.style.backgroundColor = '#0e1117';
}
</script>
""", unsafe_allow_html=True)

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
    .stButton > button {
        background: #262730;
        color: #ffffff;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background: #3a3a3a;
        border-color: #6a6a6a;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    h1 {
        text-align: center !important;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stRadio > div {
        color: #ffffff;
    }
    .stSelectbox > div > div {
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        color: #ffffff;
        background-color: #262730;
    }
    .stSlider > div > div > div {
        color: #ffffff;
    }
    p, span, div {
        color: #ffffff !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stApp > header {
        background: transparent !important;
    }
    .stFileUploader > div {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
    }
    .stFileUploader label {
        color: #ffffff !important;
    }
    .stFileUploader > div > div {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    .stFileUploader section {
        background-color: #262730 !important;
        border: 2px dashed #4a4a4a !important;
    }
    .stFileUploader section > div {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #262730 !important;
        border: 2px dashed #4a4a4a !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #ffffff !important;
    }
    .stFileUploader * {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    .stFileUploader section[data-testid="stFileUploaderDropzone"] {
        background: #262730 !important;
        border: 2px dashed #4a4a4a !important;
    }
    button[kind="primary"] {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a4a !important;
    }
    .uploadedFile {
        background-color: #262730 !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #262730 !important;
        border: 2px dashed #4a4a4a !important;
    }
    [data-testid="stFileDropzoneInstructions"] {
        color: #ffffff !important;
    }
    button[kind="secondary"] {
        background-color: #262730 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a4a !important;
    }
    small {
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #4a4a4a;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a4a4a;
        color: #ffffff;
        border-color: #6a6a6a;
    }
    .stCodeBlock {
        background-color: #262730 !important;
    }
    .stCodeBlock > div {
        background-color: #262730 !important;
    }
    pre {
        background-color: #262730 !important;
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
        
        # Load the retrained model (TF 2.20 compatible)
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    """Process uploaded image for mask detection"""
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    results = []
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face_roi = img_array[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))  # MobileNetV2 input size
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        # Predict
        prediction = model.predict(face_batch, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        mask_status = 'With Mask' if prediction > 0.5 else 'Without Mask'
        color = (0, 255, 0) if prediction > 0.5 else (255, 0, 0)
        
        # Draw bounding box
        cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
        
        # Add label
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
    st.markdown("<div style='text-align: center;'><strong>AI-powered mask detection using CNN | 99.58% Accuracy</strong></div>", unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("🚨 Model not found!")
        st.info("""
        **For Deployment:**
        1. Download `transfer_face_mask_detector.h5` from your local system
        2. Upload it to your GitHub repository
        3. Redeploy the Streamlit app
        
        **For Local Testing:**
        Ensure `transfer_face_mask_detector.h5` is in the same directory as this script.
        """)
        st.stop()
    
    st.success("Model Online!")
    
    # Main interface with custom styling
    tab1, tab2, tab3 = st.tabs(["Image Detection", "Model Info", "Live Camera"])
    
    with tab1:
        st.header("Image Detection")
        
        option = st.radio(
            "Input method:", 
            ["Upload File", "Image URL"],
            horizontal=True
        )
        
        uploaded_file = None
        image_url = None
        
        if option == "Upload File":
            st.markdown("""
            <style>
            .stFileUploader button {
                background-color: #262730 !important;
                color: #ffffff !important;
                border: 1px solid #4a4a4a !important;
            }
            </style>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Choose image", 
                type=['jpg', 'jpeg', 'png']
            )
        else:
            image_url = st.text_input(
                "Image URL",
                placeholder="https://example.com/image.jpg"
            )
        
        # Process uploaded file or URL
        image = None
        
        if uploaded_file is not None:
            # Validate file
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large. Please upload an image smaller than 10MB.")
                st.stop()
            
            # Read and open image properly
            try:
                bytes_data = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(bytes_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                st.error(f"Error loading image: {str(e)}. Please upload a valid image file.")
                st.stop()
        
        elif image_url:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                else:
                    st.error(f"Failed to load image from URL. Status code: {response.status_code}")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading image from URL: {str(e)}")
                st.stop()
        
        if image is not None:
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                # Process image
                with st.spinner("Analyzing image..."):
                    processed_img, results = process_image(image, model)
                
                # Display processed image
                st.image(processed_img, use_column_width=True)
                
                # Display results with better styling
                if results:
                    st.markdown(f"### Detection Results: {len(results)} face(s) found")
                    
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
                    st.info("No faces detected in the image. Try a different image with visible faces.")
    
    with tab2:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.code("""
Transfer Learning Architecture:
- Base: MobileNetV2 (pre-trained)
- GlobalAveragePooling2D
- Dense(128, ReLU)
- Dropout(0.5)
- Dense(1, Sigmoid) - Output Layer

Total Parameters: 2.4M
Trainable Parameters: 132K
            """)
        
        with col2:
            st.subheader("Model Performance")
            st.metric("Test Accuracy", "99.58%")
            st.metric("Input Size", "224x224x3")
            st.metric("Classes", "2 (With/Without Mask)")
            
        st.subheader("Technical Details")
        st.info("""
        **Base Model:** MobileNetV2 (ImageNet pre-trained)
        **Optimizer:** Adam (lr=0.0001)
        **Loss Function:** Binary Crossentropy
        **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        **Data Augmentation:** Rotation, Shift, Zoom, Flip
        """)
    
    with tab3:
        st.header("Camera Detection")
        
        # Camera detection settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detection Settings**")
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                0.5, 1.0, 0.8,
                help="Higher values = more strict detection"
            )
        
        with col2:
            st.markdown("**Camera Options**")
            camera_option = st.selectbox(
                "Select Camera Mode",
                ["Live Detection", "Upload from Device"]
            )
        
        if camera_option == "Live Detection":
            st.subheader("🎥 Live Camera Detection")
            
            # Auto-refresh for live detection
            if st.button("🔄 Refresh Detection", key="refresh_btn"):
                st.rerun()
            
            # Live camera capture
            camera_image = st.camera_input("Live mask detection", key="live_camera")
            
            if camera_image is not None:
                # Process the captured image
                image = Image.open(camera_image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image immediately
                with st.spinner("🔍 Analyzing live feed..."):
                    processed_img, results = process_image(image, model)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Live Feed")
                    st.image(processed_img, use_column_width=True)
                
                with col2:
                    st.subheader("Live Results")
                    
                    # Display results
                    if results:
                        for i, result in enumerate(results):
                            status = result['status']
                            confidence = result['confidence']
                            
                            if confidence >= confidence_threshold * 100:
                                if status == 'With Mask':
                                    st.markdown(f"""
                                    <div class="success-result">
                                        ✅ <strong>MASK DETECTED</strong>
                                        <span style="float: right;">{confidence:.1f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="warning-result">
                                        ⚠️ <strong>NO MASK</strong>
                                        <span style="float: right;">{confidence:.1f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("👤 No faces detected")
                
                # Auto-refresh every 2 seconds
                time.sleep(0.1)
                st.rerun()
        
        elif camera_option == "Upload from Device":
            st.subheader("📱 Upload from Device Camera")
            st.markdown("""
            <div class="detection-card">
                <h4>📷 Mobile Camera Upload</h4>
                <p>Use your device's camera to take a photo and upload it for detection.</p>
                <ol>
                    <li>Click "Browse files" below</li>
                    <li>Select "Camera" or "Take Photo" option</li>
                    <li>Capture your photo</li>
                    <li>Upload for instant mask detection</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # File uploader for mobile camera
            mobile_image = st.file_uploader(
                "Take photo with device camera",
                type=['jpg', 'jpeg', 'png'],
                help="On mobile: tap to access camera directly"
            )
            
            if mobile_image is not None:
                # Process mobile uploaded image
                try:
                    bytes_data = mobile_image.getvalue()
                    image = Image.open(io.BytesIO(bytes_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Uploaded Image")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.subheader("Detection Results")
                        
                        # Process image
                        with st.spinner("Analyzing uploaded image..."):
                            processed_img, results = process_image(image, model)
                        
                        # Display processed image
                        st.image(processed_img, use_column_width=True)
                        
                        # Display results
                        if results:
                            st.markdown(f"### Detection Results: {len(results)} face(s) found")
                            
                            for i, result in enumerate(results):
                                status = result['status']
                                confidence = result['confidence']
                                
                                if confidence >= confidence_threshold * 100:
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
                            st.info("No faces detected in the uploaded image.")
                            
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        # Deployment info
        st.markdown("""
        <div class="detection-card">
            <h4>🌐 Deployment Ready</h4>
            <p>This camera functionality works perfectly on:</p>
            <ul>
                <li>✅ Streamlit Cloud deployment</li>
                <li>✅ Mobile browsers (iOS/Android)</li>
                <li>✅ Desktop browsers</li>
                <li>✅ No hardware dependencies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()