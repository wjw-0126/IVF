#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load the model
model = joblib.load('XGBoost.pkl')
# Define feature options
cp_options = {    
    1: 'Typical angina (1)',    
    2: 'Atypical angina (2)',    
    3: 'Non-anginal pain (3)',    
    4: 'Asymptomatic (4)'}
restecg_options = {    
    0: 'Normal (0)',    
    1: 'ST-T wave abnormality (1)',    
    2: 'Left ventricular hypertrophy (2)'}
slope_options = {    
    1: 'Upsloping (1)',   
    2: 'Flat (2)',    
    3: 'Downsloping (3)'}
thal_options = {    
    1: 'Normal (1)',    
    2: 'Fixed defect (2)',    
    3: 'Reversible defect (3)'}
# Define feature names
feature_names = [    "Female age", "Primary infertility", "BMI", "FSH", "E2", "AFC"]
# Streamlit user interface
st.title("Heart Disease Predictor")
# age: numerical input
female_age = st.number_input("Female age:", min_value=18, max_value=100, value=30)
# sex: categorical selection
yuanfa = st.selectbox("Primary infertility (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# trestbps: numerical input
bmi = st.number_input("BMI:", min_value=10.00, max_value=100.00, value=24.00)
# chol: numerical input
FSH = st.number_input("FSH (U/L):", min_value=0.00, max_value=50.00, value=20.00)
# thalach: numerical input
E2 = st.number_input("E2 (pg/ml):", min_value=0.00, max_value=500.00, value=150.00)
# oldpeak: numerical input
AFC = st.number_input("AFC:", min_value=0, max_value=100, value=10)
# Process inputs and make predictions
feature_values = [female_age,yuanfa,bmi,FSH,E2,AFC]

# 创建一个新的DataFrame用于存放归一化后的结果
x = pd.DataFrame([feature_values], columns=feature_names)
# 获取连续变量的列索引
continuous_cols = [0, 2, 3, 4, 5]  # 假设BMI, FSH, E2, AFC是连续变量
# 定义归一化处理器
scaler = StandardScaler()
# 对连续变量列进行归一化处理
x.iloc[:, continuous_cols] = scaler.fit_transform(x.iloc[:, continuous_cols])

#features = np.array([feature_values])
if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(x)[0]    
    predicted_proba = model.predict_proba(x)[0]
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of heart disease. "            
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "           
            "While this is just an estimate, it suggests that you may be at significant risk. "            
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "           
            "to ensure you receive an accurate diagnosis and necessary treatment."        )   
    else:        
        advice = (            
            f"According to our model, you have a low risk of heart disease. "            
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
            "However, maintaining a healthy lifestyle is still very important. "           
            "I recommend regular check-ups to monitor your heart health, "            
            "and to seek medical advice promptly if you experience any symptoms."        )
    st.write(advice)
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(x)#x
    shap.force_plot(explainer.expected_value, shap_values[0], x, matplotlib=True)#x    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


# In[ ]:




