import streamlit as st
import numpy as nps
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


#Streamlit app
st.title("Customer Churn Prediction")

#User input
geography = st.selectbox("Geography", encoder.categories_[0])
gender=st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 100)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, step=1000.0)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])


#prepare for input data
input_data = {
    "CreditScore": [credit_score],
    "Gender": [label_encoder.transform( [gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "EstimatedSalary": [estimated_salary]

}

#one hot encoding geography
geo_encoded = encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder.get_feature_names_out(["Geography"]))

#Converting the input data to DataFrame
input_data = pd.DataFrame(input_data)


#Combinining one hot encoded geography with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scaling the input data
input_data= scaler.transform(input_data)



# Prediction button
if st.button('Predict'):
    # Assuming `model` is the trained model, for example, a logistic regression or any classifier
    prediction = model.predict(input_data)  # Predict the result
    
    # Display the result
    if prediction == 1:
        st.write("The customer will churn!")
    else:
        st.write("The customer will not churn.")