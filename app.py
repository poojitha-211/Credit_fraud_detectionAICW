import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = joblib.load("credit_card_model.pkl")
    return model

model = load_model()


st.title("💳 Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is **Fraudulent** or **Legitimate**.")

st.sidebar.header("Transaction Details")


def fill_fraud_data():
    fraud_data = {
        'V1': -2.312227, 'V2': 1.951992, 'V3': -1.609851, 'V4': 3.997906, 
        'V5': -0.522188, 'V6': -1.426545, 'V7': -2.537387, 'V8': 1.391657, 
        'V9': -2.770089, 'V10': -2.772272, 'V11': 3.202033, 'V12': -2.899907, 
        'V13': -0.595222, 'V14': -4.289254, 'V15': 0.389724, 'V16': -1.140747, 
        'V17': -2.830056, 'V18': -0.016822, 'V19': 0.416956, 'V20': 0.126911, 
        'V21': 0.517232, 'V22': -0.035049, 'V23': -0.465211, 'V24': 0.320198, 
        'V25': 0.044519, 'V26': 0.177840, 'V27': 0.261145, 'V28': -0.143276, 
        'Amount': 0.0
    }
    for k, v in fraud_data.items():
        st.session_state[k] = v

if st.sidebar.button("Test with Fraud Example"):
    fill_fraud_data()


features = [f'V{i}' for i in range(1, 29)] + ['Amount']
for f in features:
    if f not in st.session_state:
        st.session_state[f] = 0.0

V1 = st.sidebar.number_input("V1", key="V1", value=0.0)
V2 = st.sidebar.number_input("V2", key="V2", value=0.0)
V3 = st.sidebar.number_input("V3", key="V3", value=0.0)
V4 = st.sidebar.number_input("V4", key="V4", value=0.0)
V5 = st.sidebar.number_input("V5", key="V5", value=0.0)
V6 = st.sidebar.number_input("V6", key="V6", value=0.0)
V7 = st.sidebar.number_input("V7", key="V7", value=0.0)
V8 = st.sidebar.number_input("V8", key="V8", value=0.0)
V9 = st.sidebar.number_input("V9", key="V9", value=0.0)
V10 = st.sidebar.number_input("V10", key="V10", value=0.0)
V11 = st.sidebar.number_input("V11", key="V11", value=0.0)
V12 = st.sidebar.number_input("V12", key="V12", value=0.0)
V13 = st.sidebar.number_input("V13", key="V13", value=0.0)
V14 = st.sidebar.number_input("V14", key="V14", value=0.0)
V15 = st.sidebar.number_input("V15", key="V15", value=0.0)
V16 = st.sidebar.number_input("V16", key="V16", value=0.0)
V17 = st.sidebar.number_input("V17", key="V17", value=0.0)
V18 = st.sidebar.number_input("V18", key="V18", value=0.0)
V19 = st.sidebar.number_input("V19", key="V19", value=0.0)
V20 = st.sidebar.number_input("V20", key="V20", value=0.0)
V21 = st.sidebar.number_input("V21", key="V21", value=0.0)
V22 = st.sidebar.number_input("V22", key="V22", value=0.0)
V23 = st.sidebar.number_input("V23", key="V23", value=0.0)
V24 = st.sidebar.number_input("V24", key="V24", value=0.0)
V25 = st.sidebar.number_input("V25", key="V25", value=0.0)
V26 = st.sidebar.number_input("V26", key="V26", value=0.0)
V27 = st.sidebar.number_input("V27", key="V27", value=0.0)
V28 = st.sidebar.number_input("V28", key="V28", value=0.0)
Amount = st.sidebar.number_input("Transaction Amount", key="Amount", value=0.0)


FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9',
    'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
    'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]


input_data = dict.fromkeys(FEATURES, 0.0)


input_data['V1'] = V1
input_data['V2'] = V2
input_data['V3'] = V3
input_data['V4'] = V4
input_data['V5'] = V5
input_data['V6'] = V6
input_data['V7'] = V7
input_data['V8'] = V8
input_data['V9'] = V9
input_data['V10'] = V10
input_data['V11'] = V11
input_data['V12'] = V12
input_data['V13'] = V13
input_data['V14'] = V14
input_data['V15'] = V15
input_data['V16'] = V16
input_data['V17'] = V17
input_data['V18'] = V18
input_data['V19'] = V19
input_data['V20'] = V20
input_data['V21'] = V21
input_data['V22'] = V22
input_data['V23'] = V23
input_data['V24'] = V24
input_data['V25'] = V25
input_data['V26'] = V26
input_data['V27'] = V27
input_data['V28'] = V28


input_data['Amount'] = Amount

input_df = pd.DataFrame([input_data])


st.subheader("Entered Transaction Data")
st.dataframe(input_df)


if st.button("Predict Transaction Data"):

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.write("Fraud Probability:")
    st.write(f"{probability[0][1] * 100:.2f}%")






    

