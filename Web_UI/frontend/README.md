---

# 🩺 Cervical Cancer Detection Web App

A **Deep Learning powered** web application to detect cervical cancer types from uploaded images using a **DARTS (Differentiable Architecture Search)** trained model.
Built with **FastAPI** (Backend) and **React + Vite** (Frontend), featuring **light/dark mode**, **confidence visualizations**, and smooth **animations**.

---

## 📸 Demo Preview

<img width="1919" height="899" alt="image" src="https://github.com/user-attachments/assets/1887e0db-5db9-40a9-af10-e5c840b50a3c" />


---

## ✨ Features

* 📂 **Upload Images** — Predict cervical cancer type instantly.
* 🌗 **Dark/Light Mode Toggle** — Stylish slide switch with icons.
* 📊 **Confidence Progress Bar** — Animated bar showing prediction confidence.
* 🎯 **Accurate Predictions** — Powered by a trained `DARTSNetwork` model.
* 📱 **Responsive Design** — Works on desktop, tablet, and mobile screens.
* 🎨 **Modern UI/UX** — Centered layout with smooth animations.

---

## 🏗️ Tech Stack

### **Frontend**

* ⚡ [React](https://react.dev/) + [Vite](https://vitejs.dev/)
* 🎨 CSS animations & responsive design
* 🌗 Theme switching with `localStorage` persistence

### **Backend**

* 🚀 [FastAPI](https://fastapi.tiangolo.com/)
* 🔥 PyTorch for model inference
* 📦 TorchVision for preprocessing

---

## 📂 Project Structure

```
CervicalCancer/
│
├── backend/                 # FastAPI backend
│   ├── main.py               # API routes & server
│   ├── model_loader.py       # Loads trained model & predicts
│   ├── darts_final.pth       # Trained model weights
│   ├── labels.txt            # Class labels
│   ├── requirements.txt      # Backend dependencies
│
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx           # Main App component
│   │   ├── App.css           # Global styles
│   │   ├── components/
│   │   │   ├── UploadForm/   # File upload UI
│   │   │   ├── ToggleSwitch/ # Dark/Light mode toggle
│   │   │   └── PredictionResult/ # Displays results with animations
│   ├── public/
│   │   └── favicon.ico       # Custom tab icon
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

This will start FastAPI on **[http://localhost:8000](http://localhost:8000)**

---

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

This will start React on **[http://localhost:5173](http://localhost:5173)**

---

## 🔄 How It Works

1. User uploads an image via frontend.
2. Image is sent to **FastAPI** backend as `multipart/form-data`.
3. Backend loads the **trained DARTS model** and runs inference.
4. Prediction & confidence score are returned as JSON.
5. Frontend displays the result with a confidence progress bar.

---

## 🧠 Model Training

The `DARTSNetwork` model was trained on cervical cancer images using:

* Custom architecture search with **Differentiable Architecture Search**
* Dataset: `data/raw/cervicalCancer`
* Image size: `64x64`
* Optimizer: **Adam**
* Loss: **CrossEntropyLoss**
* Epochs: **30**

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Dhanush** — AI/ML Developer
📧 Email: [s.dhanush1106@gmail.com](mailto:s.dhanush1106@gmail.com)

---
