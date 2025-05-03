import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib



df=pd.read_csv('E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/data/train.csv')

print("file loaded")

#we are dropping this columns because 80% data are missing and can cause misleading predicitons

#id – Just a row identifier, not useful for modeling → Drop.
#Name – Personal identifiers, not predictive → Drop.
#Study Satisfaction, Academic Pressure, CGPA
#→ All have ~80% missing, making them unreliable → Droping all three.
#Profession - too many noisy data present and 26% missing value

df.drop(columns=['id', 'Name', 'Study Satisfaction', 'Academic Pressure', 'CGPA','Profession'], inplace=True)

# mapping

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

df['Working Professional or Student'] = df['Working Professional or Student'].map({'Student': 0,'Working Professional': 1})

# Define mapping
sleep_map = {
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '6-7 hours': 6.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9,
    '3-4 hours': 3.5,
    '4-5 hours': 4.5,
    '6-8 hours': 7,
    '8-9 hours': 8.5,
    '9-11 hours': 10,
    '10-11 hours': 10.5,
    '8 hours': 8
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)
df['Sleep Duration'] = df['Sleep Duration'].fillna(df['Sleep Duration'].mean())

valid_diet = {'Healthy': 2, 'Moderate': 1, 'Unhealthy': 0}
df['Dietary Habits'] = df['Dietary Habits'].map(valid_diet)
df['Dietary Habits'] = df['Dietary Habits'].fillna(df['Dietary Habits'].mode()[0])

df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})

df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})

top_cities = df['City'].value_counts().nlargest(30).index
df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')

top_degrees = df['Degree'].value_counts().nlargest(27).index
df['Degree'] = df['Degree'].apply(lambda x: x if x in top_degrees else 'Other')

bins = [17, 25, 35, 45, 55, 65]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
df['Age'] = df['Age_Group'].cat.codes
df.drop(columns='Age_Group',inplace=True)

# label encoding for city and degree

le= LabelEncoder()
df['City'] = le.fit_transform(df['City'])
df['Degree'] = le.fit_transform(df['Degree'])

print("encoding done")

#imputations
"""knn_imputer = KNNImputer(n_neighbors=5)
ordinal_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']  # include similar ones
df[ordinal_cols] = knn_imputer.fit_transform(df[ordinal_cols])"""

ordinal_cols = ['Work Pressure', 'Job Satisfaction', 'Financial Stress']
for i in ordinal_cols:
    df[i] = df[i].fillna(df[i].mode()[0])

print("imputations done")

cols_to_scale = [
    'Age', 'Work Pressure', 'Job Satisfaction',
    'Sleep Duration', 'Work/Study Hours', 'Financial Stress'
]

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

df.to_csv("E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/processedData/train_processedData.csv",index=False)

joblib.dump(le, 'E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/label_encoder.pkl')
#joblib.dump(knn_imputer, 'E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/knn_imputer.pkl')
joblib.dump(scaler, 'E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/scaler.pkl')

print("Preprocessing completed and saved models!")