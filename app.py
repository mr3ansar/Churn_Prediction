import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

#load the trined model
model = tf.keras.models.load_model('churn_model.h5')
# load the encoder scaler   
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('geo_encoder.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    Scaler = pickle.load(file)

st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',16,99)
balance = st.number_input('Balance')
credit_score= st.number_input('CreditScore')
estimated_salary= st.number_input('EstimatedSalary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('NumberofProducts',1,4)
has_cr_card = st.selectbox('HasCreditCard',[0,1])
is_active_member = st.selectbox('IsActiveMember',[0,1])

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data

input_data_scaled = Scaler.transform(input_data)

### Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba: .2f}")

if prediction_proba > 0.5:
    st.write('The Customer is likely to leave the bank')
else:
    st.write('The Custommer is not likely to leave the bank.')
