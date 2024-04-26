import streamlit as st
import pickle as pkl
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


def set_background(image_url):
    # Define some custom CSS to set the background image
    st.markdown(
        f"""
        <style>
        .appview-container {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .ezrtsby2{{
        display:none;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )
set_background("https://wallpapercave.com/wp/9bLG21c.jpg")

Gender_transform=pkl.load(open('../Models/Gender_transform.pkl','rb'))
Output_transform=pkl.load(open('../Models/Output_transform.pkl','rb'))
Occupation_transform=pkl.load(open('../Models/Occupation_transform.pkl','rb'))
model=pkl.load(open('../Models/Model.pkl','rb'))



    


st.title("Stress Level Predictor")

selected_option = st.selectbox(
    'Select a Gender:',
    ['Male', 'Female']
)

user_age = st.number_input('Enter your age:', min_value=1, max_value=100, value=30, step=1)


if user_age < 0 or user_age > 120:
    st.error('Please enter a valid age between 1 and 100.')
user_occpation=st.selectbox(
    "Select Your Occupation:",
    ['Nurse','Doctor','Engineer','Lawyer','Teacher','Accountant','Other']
)

user_sleep_duration = st.number_input('Enter your Sleep_duration(hour):', min_value=1.0, max_value=15.0, value=1.0, step=0.1)

user_quality_sleep = st.number_input('Enter your Quality of sleep:', min_value=1, max_value=10, value=5, step=1)



if( st.button("Submit")):
    print(selected_option,user_age,user_occpation,user_sleep_duration,user_quality_sleep)
    arr_occupation=Occupation_transform.transform([[user_occpation]])
    arr_gender=Gender_transform.transform([[selected_option]])
    arr=np.array([[user_age,user_sleep_duration,user_quality_sleep]])
    final_arr=np.hstack([arr,arr_occupation,arr_gender])    
    output=model.predict(final_arr)
    label_output=Output_transform.inverse_transform(output)
    st.success(f"Your Stress Level Is  {label_output[0][0]}")

  