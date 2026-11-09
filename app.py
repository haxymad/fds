import pickle
import pandas as pd
import streamlit as st

with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)
model, scaler, columns = saved['model'], saved['scaler'], saved['columns']

st.title('Titanic Survival Predictor')
st.write('Logistic Regression model trained on the Titanic dataset.')

pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.radio('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 28)
sibsp = st.number_input('Siblings / Spouses aboard', 0, 8, 0)
parch = st.number_input('Parents / Children aboard', 0, 6, 0)
fare = st.number_input('Fare', 0.0, 520.0, 14.5)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

row = pd.DataFrame([{
    'Pclass': pclass, 'Sex': 1 if sex == 'female' else 0, 'Age': age,
    'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
    'Embarked_Q': int(embarked == 'Q'), 'Embarked_S': int(embarked == 'S'),
}])[columns]

if st.button('Predict'):
    prob = model.predict_proba(scaler.transform(row))[0, 1]
    st.write(f'Survival probability: **{prob:.2%}**')
    st.success('Likely Survived') if prob >= 0.5 else st.error('Likely Did Not Survive')
