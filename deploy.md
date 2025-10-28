# 🚀 Deployment Guide

## Local Development

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Application**
```bash
python run.py
# OR
streamlit run live_detection.py
```

## Production Deployment

### 1. Streamlit Cloud
```bash
# Push to GitHub
git add .
git commit -m "Face mask detection app"
git push origin main

# Deploy on streamlit.io
# Connect GitHub repo and deploy
```

### 2. Docker Deployment
```bash
# Build image
docker build -t face-mask-detection .

# Run container
docker run -p 8501:8501 face-mask-detection
```

### 3. Heroku Deployment
```bash
# Install Heroku CLI
heroku create your-app-name
git push heroku main
```

### 4. AWS/GCP Deployment
```bash
# Use Docker image with cloud services
# Configure load balancer for scaling
```

## Environment Variables
```bash
export MODEL_PATH=accurate_mask_detector.h5
export PORT=8501
```

## Performance Optimization
- Use GPU for faster inference
- Implement model caching
- Optimize frame processing
- Use CDN for static assets