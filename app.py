import streamlit as st
import pickle as pk
import numpy as np
import time

pipe = pk.load(open('pipe.pkl','rb'))


def main():
    st.set_page_config(page_title="Diabeties Prediction App", page_icon="banner-250x250-1-1.jpg")
    st.title('Diabetic Prediction')
    st.divider()
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17)
    Glucose = st.number_input('Glucose', min_value=0, max_value=200)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=122)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=99)
    Insulin = st.number_input('Insulin', min_value=0, max_value=846)
    BMI = st.number_input('BMI', min_value=0.0, max_value=67.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5)
    Age = st.number_input('Age', min_value=21, max_value=81)


    if st.button('Predict'):
        query = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        query = np.asarray(query)
        query = query.reshape(1,-1)
        result = pipe.predict(query)
        class_pred = 'None'
        if result[0] == 1: 
            class_pred = 'Diabetic'
        else:
            class_pred = 'Non-Diabetic'
        with st.spinner('Making Prediction...'):
            time.sleep(5)
        st.success(f'The Patient is **{class_pred}**')
    st.divider()
    st.caption('This Project was created by **Pranjal Jais**')
    st.page_link('https://github.com/pranjal0jais/Diabeties_prediction_app',label = 'Instagram',icon = '💬')
    st.page_link('https://github.com/pranjal0jais', label = 'github', icon = '🔗')
if __name__ == '__main__':
    main()

