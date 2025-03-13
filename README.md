# 🚀 SignVision

## 📌 Project Overview

**SignVision** is an AI-powered sign language recognition system that enables real-time sign detection and translation. It utilizes **MediaPipe** for landmark detection and **machine learning models** to classify gestures. The application is built using **Tkinter** for the UI and supports **SQLite** for user authentication and data storage.

---

## 🛠 Tech Stack

### Backend:
- **Python** (Tkinter-based GUI)
- **SQLite** (User authentication & data storage)
- **MediaPipe** (Real-time gesture detection)

### Frontend:
- **Tkinter** (User-friendly graphical interface)
- **OpenCV** (Camera integration & image processing)
- **Pyttsx3** (Text-to-speech support)

### Other Tools:
- **Scikit-learn** (Machine learning for sign recognition)
- **NumPy & Joblib** (Data processing & model storage)

---

## 📌 Features
- 🔐 **User Authentication** (Register/Login using SQLite database)
- 🎥 **Real-time Sign Detection** (MediaPipe & OpenCV integration)
- 🧠 **Machine Learning Model** (Trains and detects sign actions)
- 📝 **Text Output** (Displays recognized actions as text)
- 🔊 **Text-to-Speech Support** (Reads detected actions aloud)
- 🎨 **Interactive UI** (Built using Tkinter)

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/iamaftab18/SignVision-AI-Powered-Sign-Language-Detection.git
cd SignVision-AI-Powered-Sign-Language-Detection
```

### 2️⃣ Set Up Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```sh
python app.py
```

---

## 🏗 Project Structure
```
SignVision-AI-Powered-Sign-Language-Detection/
│── models/              # Trained ML models
│── data/                # Stored sign gesture data
│── app.py               # Main application file
│── sign_language.db      # SQLite database
│── requirements.txt      # Project dependencies
│── README.md            # Project documentation
```

---

## 🎯 How It Works
1️⃣ **User registers/logs in** using the GUI.
2️⃣ **Camera captures hand and face movements** using MediaPipe.
3️⃣ **Extracted keypoints** are fed into the trained machine learning model.
4️⃣ **Recognized gestures** are displayed as text.
5️⃣ **Optional: Text-to-Speech reads out the detected actions.**

---


## 🤝 Contributing
We welcome contributions! If you'd like to improve this project, follow these steps:
1. **Fork the repository** 🍴
2. **Create a new branch** 🔀
3. **Make your changes** ✨
4. **Submit a pull request** 📩

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
👨‍💻 **Your Name**  
📧 Email: aftabsm18@gmail.com 
🔗 GitHub: [iamaftab18](https://github.com/iamaftab18)  

---

_💙 Thank you for checking out SignVision! Let's make sign language more accessible with AI!_ 🚀

