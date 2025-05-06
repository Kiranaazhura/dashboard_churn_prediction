import pandas as pd
import numpy as np

# import model final
from xgboost import XGBClassifier

# untuk load model
import pickle
import joblib

import streamlit as st

# ==========================================================================================

# judul
st.write("""
         <div style="text-align: center;">
         <h2>Customer Churn Prediction 🤑</h2>
         </div>
         <p>Model ini digunakan untuk memprediksi apakah seorang pelanggan akan berhenti menggunakan layanan 
         perbankan atau tetap menggunakan layanan perbankan.</p>
         <p>Model ini menggunakan algoritma XGBoost untuk melakukan prediksi.</p>
         """, unsafe_allow_html=True)

# sidebar menu for input
st.sidebar.header("Please Input your Customer's Feature ")

# untuk input numerik
def user_input_feature():
    cred_score = st.sidebar.slider('Credit Score', 
                    min_value = 350,
                    max_value = 850,
                    value = 500)

    balance = st.sidebar.slider(label = 'Balance',
                            min_value = 0,
                            max_value = 260000,
                            value = 130000)

    salary = st.sidebar.slider(label = 'EstimatedSalary',
                            min_value = 11,
                            max_value = 200000,
                            value = 100000)

    age = st.sidebar.number_input(label = 'Age',
                            min_value = 18,
                            max_value = 92,
                            value = 30)

    tenure = st.sidebar.number_input(label = 'Tenure',
                            min_value = 0,
                            max_value = 10,
                            value = 5)

    product = st.sidebar.number_input(label = 'NumOfProduct',
                            min_value = 1,
                            max_value = 5,
                            value = 2)

    # untuk input kategori
    cr_card = st.sidebar.selectbox(label = 'HasCrCard',
                            options = [0,1])

    active = st.sidebar.selectbox(label = 'IsActiveMember',
                            options = [0,1])

    gender = st.sidebar.selectbox(label = 'Gender',
                            options = ["Female", "Male"])

    geo = st.sidebar.selectbox(label = 'Geography',
                            options = ["France", "Germany", "Spain"])
    
    df = pd.DataFrame()
    df['CreditScore'] = [cred_score]
    df['Geography'] = [geo]
    df['Gender'] = [gender]
    df['Age'] = [age]
    df['Tenure'] = [tenure]
    df['Balance'] = [balance]
    df['NumOfProducts'] = [product]
    df['HasCrCard'] = [cr_card]
    df['IsActiveMember'] = [active]
    df['EstimatedSalary'] = [salary]

    return df

df_feature = user_input_feature()

# memanggil model
model = joblib.load('model_xgboost_joblib')

# predict
pred = model.predict(df_feature)

# untuk membuat layout menjadi 2 bagian
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Characteristics")
    st.write(df_feature.transpose())

with col2:
    st.subheader("Predicted Result")
    if pred == [1]:
        st.write("<h4 style = 'color : red;'>Your Customer is likely to CHURN😔</h4>", unsafe_allow_html=True)
    else:
        st.write("<h4 style = 'color : green;'>Your Customer is predicted to STAY🤗</h4>", unsafe_allow_html=True)


    