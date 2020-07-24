import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 



st.subheader('Predire si un patient est diabetique ou non  : ')

df=pd.read_csv('/Users/mac/Desktop/Bureau - MacBook Pro/Master IDDL/Machine Learning/Priject Diabete/diabetes.csv')

st.subheader('Data Information : ')

st.dataframe(df)
st.subheader('Data type : ')
df.dtypes
st.write(df.describe())

chart = st.bar_chart(df)


x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values

X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.25,random_state=0)


def get_user_input():
    Pregnancies =st.sidebar.slider('pregnacies',0,17,3)
    Glucose =st.sidebar.slider('Glucose',0,199,117)
    BloodPressure =st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness =st.sidebar.slider('SkinThickness',0,99,23)
    Insulin =st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI =st.sidebar.slider('BMI',0.0,67.1,32.0)
    DiabetesPedigreeFunction =st.sidebar.slider('DiabetesPedigreeFunction',0.078,2.42,0.3725)
    Age =st.sidebar.slider('Age',21,81,29)

    user_data={
        'Pregnancies':Pregnancies,
        'Glucose':Glucose,
        'BloodPressure':BloodPressure,
        'SkinThickness':SkinThickness,
        'Insulin':Insulin,
        'BMI':BMI,
        'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
        'Age':Age
            }

    features = pd.DataFrame(user_data,index=[0])
    return features



user_input =get_user_input()
st.subheader('User Input : ')
st.write(user_input)

RandomForestClassifier =RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

st.subheader('Model Test Accuracy score : ')
st.write(str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+'%')

prediction =RandomForestClassifier.predict(user_input)

st.subheader('Classification')
st.write(prediction)

