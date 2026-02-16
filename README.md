
# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
Credit card fraud is a growing problem in digital payment systems.  
This project builds a **Machine Learning–based fraud detection system** that predicts whether a transaction is **Fraudulent** or **Legitimate**.

The model is trained on a highly imbalanced dataset and deployed using a **Streamlit web application** for real-time prediction.

---

## 🎯 Objectives
- Detect fraudulent credit card transactions accurately  
- Handle highly imbalanced data  
- Apply feature scaling and machine learning models  
- Deploy the trained model using Streamlit  

---

## 📂 Dataset Description
- **Source:** Kaggle – Credit Card Fraud Detection Dataset  
- **Total Features:** 30  
- **V1 – V28:** PCA-transformed features (for data privacy)  
- **Amount:** Transaction amount  
- **Class:**  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  

> PCA was applied to protect sensitive user information.

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Seaborn  
  - Scikit-learn  
  - Streamlit  
- **Model Persistence:** Pickle  

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Preprocessing
- Removed unnecessary columns  
- Handled class imbalance  
- Applied **StandardScaler** for feature scaling  

### 2️⃣ Model Training
- Algorithms used:
  - Logistic Regression  
  - Random Forest (optional comparison)  
- Evaluated using:
  - Precision  
  - Recall  
  - F1-Score  

### 3️⃣ Model Saving
- Trained model saved as `model.pkl`  
- Scaler saved as `scaler.pkl`  

---

## 🌐 Streamlit Web Application
The Streamlit app allows users to:
- Enter transaction details  
- Predict fraud in real time  
- View fraud probability  

### Input Features:
- V1 – V5 (sample PCA features for demo)  
- Transaction Amount  

> Remaining PCA features (V6–V28) are auto-filled with zero values to maintain feature consistency.

---

## ▶️ How to Run the Project
