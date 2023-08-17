

import streamlit as st
import joblib
import pandas as pd

st.write("# School Dropout Prediction")

col1, col2 = st.columns(2)

# getting user input
curricular_units_1st_sem_credited	= col1.number_input("Enter your curricular_units_1st_sem_credited")
curricular_units_1st_sem_enrolled	= col1.number_input("Enter your curricular_units_1st_sem_enrolled")
curricular_units_1st_sem_evaluations = col1.number_input("Enter your curricular_units_1st_sem_evaluations")	
curricular_units_1st_sem_approved	= col1.number_input("Enter your curricular_units_1st_sem_approved")
curricular_units_1st_sem_grade	    = col1.number_input("Enter your curricular_units_1st_sem_grade")
curricular_units_1st_sem_without_evaluations = col1.number_input("Enter your curricular_units_1st_sem_without_evaluations") 
curricular_units_2nd_sem_credited	= col2.number_input("Enter your curricular_units_2nd_sem_credited")
curricular_units_2nd_sem_enrolled	= col2.number_input("Enter your curricular_units_2nd_sem_enrolled")
curricular_units_2nd_sem_evaluations = col2.number_input("Enter your curricular_units_2nd_sem_evaluations")
curricular_units_2nd_sem_approved = col2.number_input("Enter your curricular_units_2nd_sem_approved")
curricular_units_2nd_sem_grade = col2.number_input("Enter your curricular_units_2nd_sem_grade")
curricular_units_2nd_sem_without_evaluations = col2.number_input("Enter your curricular_units_2nd_sem_without_evaluations")


df_pred = pd.DataFrame([[curricular_units_1st_sem_credited,curricular_units_1st_sem_enrolled,curricular_units_1st_sem_evaluations,curricular_units_1st_sem_approved,curricular_units_1st_sem_grade,curricular_units_1st_sem_without_evaluations,curricular_units_2nd_sem_credited,curricular_units_2nd_sem_enrolled,curricular_units_2nd_sem_evaluations,curricular_units_2nd_sem_approved,curricular_units_2nd_sem_grade,	curricular_units_2nd_sem_without_evaluations]],

columns= ['curricular_units_1st_sem_credited','curricular_units_1st_sem_enrolled','curricular_units_1st_sem_evaluations','curricular_units_1st_sem_approved','curricular_units_1st_sem_grade','curricular_units_1st_sem_without_evaluations','curricular_units_2nd_sem_credited','curricular_units_2nd_sem_enrolled','curricular_units_2nd_sem_evaluations','curricular_units_2nd_sem_approved','curricular_units_2nd_sem_grade','	curricular_units_2nd_sem_without_evaluations'])


rfc = joblib.load('schoolDt_rfc_model.pkl')
prediction = rfc.predict(df_pred)

if st.button('Predict'):

    if(prediction[0]==0):
        st.write('<p class="big-font">This student is likely to dropout.</p>',unsafe_allow_html=True)

    else:
        st.write('<p class="big-font">This student is likely to graduate.</p>',unsafe_allow_html=True)

        