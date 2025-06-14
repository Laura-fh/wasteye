# WastEye AI: Computer Vision for Smarter Recycling

**WastEye AI** is an end-to-end machine learning application designed to detect and classify waste materials in real time using computer vision. The goal of this project is to support better recycling practices by providing an accessible tool for accurate waste identification.

---

## 🚀 Project Overview

This project was developed as part of a data science and AI bootcamp and focuses on deploying a full-stack computer vision app. WastEye AI classifies waste items using a YOLOv10 model and serves predictions via a user-friendly interface.

---

## 🧠 Key Features

- **Image Classification:** Detects and classifies waste across multiple categories using YOLOv10  
- **Real-Time Predictions:** Deployed Streamlit interface for immediate image upload and prediction  
- **Cloud-Native Deployment:** API served via Docker and Google Cloud Run  
- **Scalable Infrastructure:** Data stored in Google Cloud Storage, training performed in Google Colab  

---

## 🛠️ Tech Stack

- **Data Exploration & Preprocessing:** Python, Pandas, NumPy, Matplotlib, Seaborn  
- **Model Training:** YOLOv10b, Google Colab, Grid Search  
- **API Development:** FastAPI, Uvicorn  
- **Deployment:** Docker, Google Cloud Run  
- **Frontend:** Streamlit (deployed on Streamlit Cloud)  
- **Cloud Storage:** Google Cloud Storage  

---

## 📊 Model Performance

- Trained on 14,000 images  
- Hyperparameter tuning using grid search  
- Achieved average precision score of **0.75 per class**

---

## 💻 How to Use

1. Upload an image via the Streamlit app: https://wasteye.streamlit.app/  
2. The model detects and classifies the waste type  
3. Prediction results are displayed with bounding boxes and confidence scores  

---

## 🌍 Impact

Wasteye aims to contribute to waste management by assisting individuals and organisations in sorting recyclables more accurately and efficiently.
