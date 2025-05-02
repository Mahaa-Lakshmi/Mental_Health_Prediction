import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
from torch import nn
from io import StringIO

PRED_MODEL_PATH = "E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/prediction_model.pth"

# Set page config
st.set_page_config(
    page_title="Depression Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    # Define the model architecture (same as training)
    class DepressionPredictor(nn.Module):
        def __init__(self, input_size):
            super(DepressionPredictor, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3),
                
                nn.Linear(32, 16),
                nn.ReLU(),
                
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)

    # Initialize and load model
    #preprocessor = load_preprocessor()
    dummy_df = pd.DataFrame(columns=[
        'Age', 'Gender', 'Sleep_Hours', 'Exercise_Frequency', 'Social_Support',
        'Stress_Level', 'BMI', 'Family_History', 'Chronic_Illness', 'Substance_Use',
        'Work_Hours', 'Education_Level', 'Income', 'Urban_Rural', 'Marital_Status'
    ])
    input_size = 10
    model = DepressionPredictor(input_size)
    model.load_state_dict(torch.load(PRED_MODEL_PATH))
    model.eval()
    return model

#preprocessor = load_preprocessor()
model = load_model()

# Prediction function
def predict_depression(input_data):
    # Preprocess input
    input_preprocessed = 10
    input_tensor = torch.tensor(
        input_preprocessed.toarray() if hasattr(input_preprocessed, 'toarray') 
        else input_preprocessed, 
        dtype=torch.float32
    )
    
    # Make prediction
    with torch.no_grad():
        predictions = model(input_tensor).numpy().flatten()
    
    return predictions

le = joblib.load('label_encoder.pkl')
knn_imputer = joblib.load('knn_imputer.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title('🧠 Depression Prediction from Mental Health Survey')

st.markdown("""
This app predicts the likelihood of depression based on mental health survey data using a deep learning model.
You can either upload a CSV file with multiple records or enter values manually for a single prediction.
""")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📤 Upload CSV File", "✍️ Manual Input"])

with tab1:
    st.header("Upload CSV File")
    st.markdown("""
    Upload a CSV file containing mental health survey data. The file should include the following columns:
    - Age (number)
    - Gender (Male/Female/Other)
    - Sleep_Hours (number)
    - Exercise_Frequency (Never/Rarely/Monthly/Weekly/Daily)
    - Social_Support (1-10)
    - Stress_Level (1-10)
    - BMI (number)
    - Family_History (No/Yes/Not sure)
    - Chronic_Illness (No/Yes)
    - Substance_Use (Never/Occasionally/Regularly)
    - Work_Hours (number)
    - Education_Level (High School/Bachelor/Master/PhD)
    - Income (Low/Medium/High)
    - Urban_Rural (Urban/Suburban/Rural)
    - Marital_Status (Single/Married/Divorced/Widowed)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Show the uploaded data
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Check if all required columns are present
            required_columns = [
                'Age', 'Gender', 'Sleep_Hours', 'Exercise_Frequency', 'Social_Support',
                'Stress_Level', 'BMI', 'Family_History', 'Chronic_Illness', 'Substance_Use',
                'Work_Hours', 'Education_Level', 'Income', 'Urban_Rural', 'Marital_Status'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("Predict for Uploaded Data"):
                    with st.spinner('Making predictions...'):
                        # Make predictions
                        predictions = predict_depression(df)
                        
                        # Add predictions to dataframe
                        result_df = df.copy()
                        result_df['Depression_Probability'] = predictions
                        result_df['Depression_Risk'] = np.where(predictions > 0.5, 'High Risk', 'Low Risk')
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(result_df)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name='depression_predictions.csv',
                            mime='text/csv'
                        )
                        
                        # Show summary statistics
                        st.subheader("Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Records", len(result_df))
                            st.metric("High Risk Cases", f"{sum(predictions > 0.5)} ({sum(predictions > 0.5)/len(predictions):.1%})")
                        
                        with col2:
                            st.metric("Average Probability", f"{np.mean(predictions):.1%}")
                            st.metric("Highest Probability", f"{np.max(predictions):.1%}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab2:
    st.header("Manual Data Entry")
    st.markdown("Please fill in the form below and click 'Predict' to see the results.")
    
    with st.form("manual_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.slider('Age', 18, 100, 30)
            gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
            marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
            
            st.subheader("Physical Health")
            bmi = st.slider('BMI', 15.0, 40.0, 22.0, step=0.1)
            sleep_hours = st.slider('Average sleep hours per night', 3, 12, 7)
            chronic_illness = st.selectbox('Chronic Illness', ['No', 'Yes'])
            
        with col2:
            st.subheader("Mental Health Factors")
            stress_level = st.slider('Perceived stress level (1-10)', 1, 10, 5)
            social_support = st.slider('Perceived social support (1-10)', 1, 10, 5)
            family_history = st.selectbox('Family History of Depression', ['No', 'Yes', 'Not sure'])
            substance_use = st.selectbox('Substance Use', ['Never', 'Occasionally', 'Regularly'])
            
            st.subheader("Lifestyle Factors")
            exercise_freq = st.selectbox('Exercise Frequency', ['Never', 'Rarely', 'Monthly', 'Weekly', 'Daily'])
            work_hours = st.slider('Weekly Work Hours', 0, 80, 40)
        
        st.subheader("Socioeconomic Factors")
        col3, col4, col5 = st.columns(3)
        with col3:
            education_level = st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD'])
        with col4:
            income = st.selectbox('Income Level', ['Low', 'Medium', 'High'])
        with col5:
            urban_rural = st.selectbox('Living Area', ['Urban', 'Suburban', 'Rural'])
        
        submitted = st.form_submit_button("Predict Depression Risk")

    if submitted:
        # Create dataframe from inputs
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Sleep_Hours': [sleep_hours],
            'Exercise_Frequency': [exercise_freq],
            'Social_Support': [social_support],
            'Stress_Level': [stress_level],
            'BMI': [bmi],
            'Family_History': [family_history],
            'Chronic_Illness': [chronic_illness],
            'Substance_Use': [substance_use],
            'Work_Hours': [work_hours],
            'Education_Level': [education_level],
            'Income': [income],
            'Urban_Rural': [urban_rural],
            'Marital_Status': [marital_status]
        })
        
        # Show the input data
        st.subheader("Your Input Data")
        st.dataframe(input_data)
        
        # Make prediction
        with st.spinner('Analyzing your data...'):
            prediction = predict_depression(input_data)[0]
        
        # Display results
        st.subheader("Prediction Result")
        
        if prediction > 0.5:
            st.error(f'High risk of depression detected ({prediction:.1%} probability)')
            st.markdown("""
            **Recommendations:**
            - Consider consulting a mental health professional
            - Reach out to friends or family for support
            - Practice self-care and stress management techniques
            """)
        else:
            st.success(f'Low risk of depression detected ({prediction:.1%} probability)')
            st.markdown("""
            **Recommendations:**
            - Maintain your healthy habits
            - Continue monitoring your mental wellbeing
            - Stay connected with your support network
            """)
        
        # Visualize the probability
        st.progress(int(prediction * 100))
        st.caption(f"Depression Risk Score: {prediction:.1%}")

# Add resources section in sidebar
with st.sidebar:
    st.header("Mental Health Resources")
    st.markdown("""
    **If you or someone you know is struggling:**
    - National Suicide Prevention Lifeline: 1-800-273-8255
    - Crisis Text Line: Text HOME to 741741
    - [NAMI Helpline](https://www.nami.org/help)
    - [Mental Health America](https://www.mhanational.org/)
    """)
    
    st.header("About This App")
    st.markdown("""
    This app uses a deep learning model trained on mental health survey data to predict depression risk.
    
    **Disclaimer:** This tool is not a substitute for professional medical advice, diagnosis, or treatment.
    """)

# Add some styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .st-b7 {
        color: white;
    }
    .st-c0 {
        background-color: #0e1117;
    }
    .css-1aumxhk {
        background-color: #0e1117;
        background-image: none;
    }
</style>
""", unsafe_allow_html=True)