import streamlit as st 
import numpy as np
import pandas as pd
import pickle
st.title("Tatanic Prediction")
df=pd.read_csv("train.csv")
pipe=pickle.load(open("pipe.pkl","rb"))


pclass=st.selectbox("Select the passenger class ",df['Pclass'].unique())
gender=st.selectbox("Select the gender of passenger ",df['Sex'].unique())
age=st.number_input("Enter the age of passenger ")
sibsp=st.number_input("how many sibsp are travel ")
parch=st.number_input("how many parch are travel ")
Fare=st.number_input("Enter the fare of passenger ")
Embarked=st.selectbox("Select the Embarked of passenger ",df['Embarked'].unique())

if st.button("predict"):
    query=np.array([pclass,gender,age,sibsp,parch,Fare,Embarked])
    query=query.reshape(1,7)
    if pipe.predict(query)==0:
        st.title("Die")
    else:
        st.title("Survived")