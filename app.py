import streamlit as st
import pickle as pk
import numpy as np
import time

# Load the trained model pipeline
pipe = pk.load(open('pipe.pkl','rb'))

# Function to define the main structure of the Streamlit app
def main():
    # Setting page title and icon
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="banner-250x250-1-1.jpg")
    
    # App title
    st.title('Diabetes Prediction')
    st.divider()
    
    # Input fields for user to enter data
    Pregnancies = st.number_input('Pregnancies', min_value=0, max_value=17)
    Glucose = st.number_input('Glucose', min_value=0, max_value=200)
    BloodPressure = st.number_input('Blood Pressure', min_value=0, max_value=122)
    SkinThickness = st.number_input('Skin Thickness', min_value=0, max_value=99)
    Insulin = st.number_input('Insulin', min_value=0, max_value=846)
    BMI = st.number_input('BMI', min_value=0.0, max_value=67.1)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5)
    Age = st.number_input('Age', min_value=21, max_value=81)

    # Prediction button
    if st.button('Predict'):
        # Prepare user input as numpy array
        query = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        query = query.reshape(1, -1)  # Reshape for single sample prediction
        
        # Predict using the loaded model pipeline
        result = pipe.predict(query)
        
        # Interpret prediction result
        if result[0] == 1: 
            class_pred = 'Diabetic'
        else:
            class_pred = 'Non-Diabetic'
        
        # Display prediction result
        with st.spinner('Making Prediction...'):
            time.sleep(2)  # Simulate prediction time
        st.success(f'The patient is **{class_pred}**')
    
    # Divider and attribution
    st.divider()
    st.caption('This project was created by **Pranjal Jais**')
    
    # Links to social profiles
    st.markdown('### Connect with Pranjal Jais:')
    st.markdown('[Instagram](https://www.instagram.com/pranjaljais13/)', unsafe_allow_html=True)
    st.markdown('[GitHub](https://github.com/pranjal0jais)', unsafe_allow_html=True)

# Entry point of the script
if __name__ == '__main__':
    main()
