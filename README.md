
# 🚦 Smart Traffic Management System with Emergency Vehicle Detection

## 📌 Overview
This project presents an AI-based Smart Traffic Management System that detects vehicles in real-time, identifies emergency vehicles, predicts future traffic, and dynamically adjusts traffic signals to reduce congestion and improve emergency response.

---

## 🎯 Key Features
- Real-time vehicle detection using YOLOv8  
- Emergency vehicle classification using EfficientNet-B0  
- Traffic prediction using LSTM (Time-Series Model)  
- Dynamic traffic signal control  
- Web application using Flask  
- Real-time visualization of traffic data  

---

## 🧠 Technologies Used
- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- TensorFlow / Keras  
- EfficientNet-B0  
- LSTM (RNN)  
- Flask  
- NumPy, Pandas, Matplotlib  

---

## ⚙️ System Architecture
Camera Input → YOLOv8 Detection → Vehicle Cropping → EfficientNet Classification → Vehicle Counting → LSTM Prediction → Signal Control → Web Dashboard

---

## 🔍 How It Works

### 1. Vehicle Detection
YOLOv8n model is used to detect vehicles like cars, buses, trucks, and bikes from real-time video.

### 2. Emergency Vehicle Detection
Detected vehicles are cropped and passed to EfficientNet-B0 model to classify:
- Ambulance  
- Fire Truck  
- Other  

### 3. Traffic Prediction
LSTM model takes last 10 vehicle counts and predicts next hour traffic.

### 4. Signal Control Logic
- Emergency detected → Priority signal (green)  
- High traffic → Increase signal time  
- Normal traffic → Default timing  

---

## 📂 Project Structure
├── app.py  
├── model_predict2.py  
├── predicter.py  
├── traffic_rnn_model.h5  
├── efficientnet.h5  
├── static/  
├── templates/  
├── notebook.ipynb  
└── README.md  

---

## ▶️ How to Run

1. Clone the repository  
git clone [https://github.com/kushalTN/Smart-Traffic-Management-System-for-Urban-Congestion] 

2. Navigate to project folder  
cd traffic-management-system  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the application  
python app.py  

5. Open in browser  
http://127.0.0.1:5000  

---

## 📊 Results
- High accuracy in emergency vehicle detection  
- ~97% accuracy in traffic prediction  
- Real-time performance using YOLOv8  
- Dynamic signal control reduces congestion  

---

## ⚠️ Limitations
- Depends on camera quality  
- Performance may reduce in low light or bad weather  
- Requires GPU for better performance  

---

## 🚀 Future Improvements
- Multi-camera system  
- Smart city integration  
- Edge deployment  
- Advanced AI-based signal optimization  

---

## 🙌 Acknowledgement
This project was developed as part of a final year academic project.

---

## 👨‍💻 Author
T.N.Kushal, Boya Valmiki Karthikeya, Sannapareddy Manohar Reddy
