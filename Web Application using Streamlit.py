import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
st.write("""
# WINE QUALITY PREDICTION

Consider that there is a wine manufacturing company and this company wants to create a new segment of wine. They want us to find the quality of the wine using several parameters. Hence the objective is to take parameters like fixed_acidity, citric_acid etc. and predict the quality of wine.
""")
st.sidebar.header('User input parameters')
# Function to store user input values
def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.6, 15.9, 7.9)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.12, 1.58, 0.52)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.26)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.90, 15.50, 2.20)
    chlorides = st.sidebar.slider('Chlorides', 0.012, 0.611, 0.079)
    free_sulphar_dioxide = st.sidebar.slider('Free Sulphar Dioxide', 1.0, 72.0, 14.0)
    total_sulphar_dioxide = st.sidebar.slider('Total Sulphar Dioxide', 6.0, 289.0, 38.0)
    density = st.sidebar.slider('Density', 0.990, 0.996, 1.003)
    pH = st.sidebar.slider('pH', 2.74, 3.31, 4.01)
    sulphates = st.sidebar.slider('Sulphates', 0.33, 0.620, 2.000)
    alcohol = st.sidebar.slider('Alcohol', 8.40, 10.20, 14.90)
    
    data = {'fixed_acidity':fixed_acidity,
           'volatile_acidity':volatile_acidity,
           'citric_acid':citric_acid,
           'residual_sugar':residual_sugar,
           'chlorides':chlorides,
           'free_sulphar_dioxide':free_sulphar_dioxide,
           'total_sulphar_dioxide':total_sulphar_dioxide,
           'density':density,
           'pH':pH,
           'sulphates':sulphates,
           'alcohol':alcohol}
    
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
# Displaying original Dataset
st.subheader('Original Dataset')
dataset = pd.read_csv('winequality.csv')
st.write(dataset)
# Displaying user input values
st.subheader('User Input Parameters')
st.write(df)
# Reshaping the dataset
X = dataset.drop('quality', axis=1)
X_2 = X.columns = [None]*len(X.columns)
Y = dataset['quality']
Y.name = None
# Dividing the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
# Training the model
model = RandomForestClassifier()
model.fit(X_train, Y_train)
# Making prediction using model
input_data_as_numpy_array = np.asarray(df)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
st.write("The quality rating of wine is : ", prediction[0])
if prediction[0] < 7:
    st.write("***The Quality of wine is BAD***")
else:
    st.write("***The Quality of wine is GOOD***")