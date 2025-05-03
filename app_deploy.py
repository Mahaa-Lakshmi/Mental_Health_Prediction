import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from torch import nn
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Mental Health Depression Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load pre-trained components
@st.cache_resource
def load_models():
    print("Inside load model function")
    le = joblib.load('models/label_encoder.pkl')
    #knn_imputer = joblib.load('E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/knn_imputer.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("returning load model function")
    #return le, knn_imputer, scaler
    return le, scaler

# Define the model architecture (must match training)
class DepressionPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DepressionPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(40),

            nn.Linear(40, 50),
            nn.LeakyReLU(),

            nn.Linear(50, 30),
            nn.LeakyReLU(),

            nn.Linear(30, 25),
            nn.LeakyReLU(),

            nn.Linear(25, 20),
            nn.LeakyReLU(0.2),

            nn.Linear(20, 10),
            nn.LeakyReLU(0.3),

            nn.Linear(10, 5),
            nn.LeakyReLU(0.3),

            nn.Dropout(0.3),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_prediction_model(path, input_size):
    print("Inside load prediction model function")
    model = DepressionPredictor(input_size)
    # Load with map_location to handle device mismatch if needed
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("returning load prediction model function")
    return model

# Load all models
try:
    #le, knn_imputer, scaler = load_models()
    le, scaler = load_models()
    # Update input_size to match your actual model input size
    model = load_prediction_model(
        "models/prediction_model.pth",
        input_size=10  # This should match your model's expected input size
    )
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Preprocessing function
def safe_label_encode(series, le):
    """Handle unseen labels by converting them to 'Other'"""
    print("inside safe_label_encode function")
    series = series.astype(str)
    known_labels = set(le.classes_)
    series = series.apply(lambda x: x if x in known_labels else 'Other')
    print("returning safe_label_encode function and the series-", series)
    return le.transform(series)

def preprocess_input(df,type='file'):
    print("inside preprocess_input function")
    # Create a copy
    processed_df = df.copy()

    print("processed df",processed_df.head(1))
    
    # Drop non-predictive columns
    processed_df.drop(columns=['id', 'Name', 'Profession'], inplace=True, errors='ignore')
    
    # Binary mappings
    processed_df['Gender'] = processed_df['Gender'].map({'Male': 0, 'Female': 1})
    processed_df['Working Professional or Student'] = processed_df['Working Professional or Student'].map(
        {'Student': 0, 'Working Professional': 1})
    processed_df['Have you ever had suicidal thoughts ?'] = processed_df['Have you ever had suicidal thoughts ?'].map(
        {'No': 0, 'Yes': 1})
    processed_df['Family History of Mental Illness'] = processed_df['Family History of Mental Illness'].map(
        {'No': 0, 'Yes': 1})
    
    # Sleep duration mapping
    sleep_map = {
        'Less than 5 hours': 4, '5-6 hours': 5.5, '6-7 hours': 6.5,
        '7-8 hours': 7.5, 'More than 8 hours': 9, '3-4 hours': 3.5,
        '4-5 hours': 4.5, '6-8 hours': 7, '8-9 hours': 8.5,
        '9-11 hours': 10, '10-11 hours': 10.5, '8 hours': 8
    }
    processed_df['Sleep Duration'] = processed_df['Sleep Duration'].map(sleep_map)
    
    # Dietary habits mapping
    diet_map = {'Healthy': 2, 'Moderate': 1, 'Unhealthy': 0}
    processed_df['Dietary Habits'] = processed_df['Dietary Habits'].map(diet_map)

    print("mapping done")
    
    # Age binning
    bins = [17, 25, 35, 45, 55, 65]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
    processed_df['Age_Group'] = pd.cut(processed_df['Age'], bins=bins, labels=labels)
    processed_df['Age'] = processed_df['Age_Group'].cat.codes
    processed_df.drop(columns='Age_Group', inplace=True)

    print("age binngin done")
    
    # Handle categoricals with safe encoding
    processed_df['City'] = safe_label_encode(processed_df['City'], le)
    processed_df['Degree'] = safe_label_encode(processed_df['Degree'], le)

    print("city and degree encoding done")
    
    # Impute missing values
    #ordinal_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']
    #processed_df[ordinal_cols] = knn_imputer.transform(processed_df[ordinal_cols])

    if type=="file":        
        ordinal_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']
        for i in ordinal_cols:
            processed_df[i] = processed_df[i].fillna(df[i].mode()[0])

    print("imputation done")
    
    # Scale numerical features
    cols_to_scale = [
        'Age', 'Work Pressure', 'Job Satisfaction',
        'Sleep Duration', 'Work/Study Hours', 'Financial Stress'
    ]
    processed_df[cols_to_scale] = scaler.transform(processed_df[cols_to_scale])
    
    print("pre-processing done")

    selected_cols= ['Age', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?', 'Job Satisfaction', 'Financial Stress', 'Work Pressure', 'Work/Study Hours', 'Dietary Habits', 'Degree', 'Sleep Duration']

    return processed_df[selected_cols]

# Prediction function
def predict_depression(input_data,type="file"):
    # Preprocess input
    print("inside predict_depression func")
    processed_data = preprocess_input(input_data,type)
    input_tensor = torch.tensor(processed_data.values, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        predictions = model(input_tensor).numpy().flatten()

    
    print("returning predict_depression func")
    return predictions

# Streamlit app
st.title('🧠 Mental Health Depression Prediction')

# Create tabs
tab1, tab2 = st.tabs(["📤 Upload CSV", "✍️ Manual Input"])

with tab1:
    st.header("CSV File Upload")
    st.markdown("""
    Upload a CSV file containing mental health survey data with the following columns:
    - id, Name, Gender, Age, City, Working Professional or Student, Profession
    - Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction
    - Sleep Duration, Dietary Habits, Degree
    - Have you ever had suicidal thoughts ?, Work/Study Hours, Financial Stress
    - Family History of Mental Illness
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            if st.button("Predict for Uploaded Data"):
                with st.spinner('Processing...'):
                    predictions = predict_depression(df)
                    
                    result_df = df.copy()
                    result_df['Depression Risk'] = np.where(predictions > 0.5, 'High Risk', 'Low Risk')
                    result_df['Probability'] = predictions
                    
                    st.subheader("Prediction Results")
                    st.dataframe(result_df[['Name', 'Depression Risk', 'Probability']])
                    
                    # Download results
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name='depression_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Show summary
                    st.subheader("Summary")
                    col1, col2 = st.columns(2)
                    col1.metric("Total Records", len(result_df))
                    col2.metric("High Risk Cases", f"{sum(predictions > 0.5)} ({sum(predictions > 0.5)/len(predictions):.1%})")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab2:
    st.header("Manual Data Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            name = st.text_input("Name")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 65, 30)
            city = st.text_input("City", "Bangalore")
            profession_type = st.selectbox(
                "Working Professional or Student",
                ["Working Professional", "Student"]
            )
            profession = st.text_input("Profession (if working)", "")
           
        
        with col2:
            st.subheader("Mental Health Factors")
            work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 5)
            job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 5)
            suicidal_thoughts = st.selectbox(
                "Have you ever had suicidal thoughts?",
                ["No", "Yes"]
            )
            family_history = st.selectbox(
                "Family History of Mental Illness",
                ["No", "Yes"]
            )
            financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 5)
        
        st.subheader("Lifestyle Factors")
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "6-7 hours", 
             "7-8 hours", "More than 8 hours"]
        )
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ["Healthy", "Moderate", "Unhealthy"]
        )
        degree = st.text_input("Degree (if student)", "")
        work_study_hours = st.slider("Work/Study Hours per day", 0, 12, 8)
        
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'id': 0,
            'Name': name,
            'Gender': gender,
            'Age': age,
            'City': city,
            'Working Professional or Student': profession_type,
            'Profession': profession,
            'Academic Pressure': np.nan,
            'Work Pressure': work_pressure,
            'CGPA': np.nan,
            'Study Satisfaction': np.nan,
            'Job Satisfaction': job_satisfaction,
            'Sleep Duration': sleep_duration,
            'Dietary Habits': dietary_habits,
            'Degree': degree,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': work_study_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }])
        
        with st.spinner('Analyzing your data...'):
            prediction = predict_depression(input_data,"form")[0]
        
        st.subheader("Result")
        if prediction > 0.5:
            st.error(f"High risk of depression ({prediction:.1%} probability)")
            st.markdown("""
            **Recommendations:**
            - Consider consulting a mental health professional
            - Reach out to friends or family for support
            - Practice self-care and stress management techniques
            """)
        else:
            st.success(f"Low risk of depression ({prediction:.1%} probability)")
            st.markdown("""
            **Recommendations:**
            - Maintain your healthy habits
            - Continue monitoring your mental wellbeing
            - Stay connected with your support network
            """)
        
        # Visualize probability
        st.progress(int(prediction * 100))
        st.caption(f"Depression Risk Score: {prediction:.1%}")

# Resources sidebar
with st.sidebar:
    st.header("Mental Health Resources")
    st.markdown("""
    **If you need help:**
    - National Suicide Prevention Lifeline: 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    - [NAMI Helpline](https://www.nami.org/help)
    """)
    
    st.header("About This App")
    st.markdown("""
    This app predicts depression risk using a deep learning model trained on mental health survey data.
    
    **Disclaimer:** This tool is not a substitute for professional medical advice.
    """)