#!/usr/bin/env python3
"""
Production runner for Face Mask Detection App
"""
import subprocess
import sys
import os

def check_model():
    """Check if model file exists"""
    if not os.path.exists('accurate_mask_detector.h5'):
        print("❌ Model file 'accurate_mask_detector.h5' not found!")
        print("Please run 'python accurate_model.py' to train the model first.")
        return False
    return True

def run_app():
    """Run the Streamlit app"""
    if not check_model():
        sys.exit(1)
    
    print("🚀 Starting Face Mask Detection App...")
    print("📱 Access the app at: http://localhost:8501")
    
    try:
        subprocess.run([
            "streamlit", "run", "live_detection.py",
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    run_app()