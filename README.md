
# 🧠 Mental Health Depression Prediction

A Streamlit web application to predict depression risk from mental health survey data using a deep learning model built with PyTorch.

## 📌 Project Overview

This project is aimed at predicting whether an individual is at high or low risk of depression based on lifestyle, demographic, and mental health-related survey responses. It leverages a trained deep neural network, extensive preprocessing, and offers two modes of input: CSV upload and manual form entry.

## 🧪 Tech Stack

- **Frontend/UI:** Streamlit
- **Backend/Model:** PyTorch
- **Preprocessing:** Pandas, Scikit-learn, LabelEncoder
- **Deployment:** Localhost (can be hosted via Streamlit Sharing or Hugging Face Spaces)

## 🧠 Model Architecture

The model is a multi-layer fully connected neural network with:
- LeakyReLU activations
- Batch Normalization
- Dropout
- Final Sigmoid layer for binary classification

## 📁 Directory Structure

```
Mental_Health_Prediction/
├── models/
│   ├── prediction_model.pth
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── app.py
└── README.md
```

## 📝 Features

- 📤 Upload CSV files for bulk prediction
- ✍️ Manual form for single data entry
- 💾 Downloadable prediction results
- 📊 Summary metrics of total and high-risk predictions

## 📂 Input Format

Ensure your CSV has the following columns:

```
id, Name, Gender, Age, City, Working Professional or Student, Profession,
Job Satisfaction, Work Pressure, Sleep Duration, Dietary Habits, Degree,
Have you ever had suicidal thoughts ?, Work/Study Hours, Financial Stress,
Family History of Mental Illness
```

## 🚀 Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure the model and preprocessing artifacts (`.pth`, `.pkl` files) are in the `models/` directory.

deployed app link :- https://mentalhealthprediction-un5ihnngkwsk4fmjmmkqwx.streamlit.app/




