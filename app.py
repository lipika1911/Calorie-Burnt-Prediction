import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load data (replace with your actual paths)
calories = pd.read_csv('./calories.csv')
exercise_data = pd.read_csv('./exercise.csv')

# Combine data
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

# Title for the app
st.title('Calories Burnt Prediction App')

# Display raw data if needed
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(calories_data)

# Data preprocessing
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model training
model = XGBRegressor()
model.fit(X_train, Y_train)

# Sidebar for user input
st.sidebar.header('Enter Parameters')

age = st.sidebar.slider('Age', min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox('Gender', ['male', 'female'])
height = st.sidebar.slider('Height', min_value=100, max_value=250, value=170)
weight = st.sidebar.slider('Weight', min_value=30, max_value=200, value=70)
duration = st.sidebar.slider('Duration (mins)', min_value=10, max_value=120, value=30)
heart_rate = st.sidebar.slider('Heart Rate', min_value=50, max_value=200, value=100)
body_temp = st.sidebar.slider('Body Temperature', min_value=35.0, max_value=40.0, value=36.5)

gender_numeric = 0 if gender == 'male' else 1

# Create input data for prediction
input_data = pd.DataFrame({
    'Gender': [gender_numeric],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'Duration': [duration],
    'Heart_Rate': [heart_rate],
    'Body_Temp': [body_temp]
}, columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])

# Make prediction
prediction = model.predict(input_data)[0]

# Display prediction
st.subheader('Calories Burnt Prediction')
st.write(f'The predicted calories burnt: {prediction:.2f} calories')

# Data Visualization for Prediction
st.subheader('Visualizations for Prediction')

# Distribution plots of Age, Height, Weight, Duration, Heart Rate, Body Temperature
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

sns.histplot(calories_data['Age'], kde=True, ax=axes[0, 0])
axes[0, 0].axvline(x=age, color='r', linestyle='--', label='User Input')
axes[0, 0].legend()
axes[0, 0].set_title('Age Distribution')

sns.histplot(calories_data['Height'], kde=True, ax=axes[0, 1])
axes[0, 1].axvline(x=height, color='r', linestyle='--', label='User Input')
axes[0, 1].legend()
axes[0, 1].set_title('Height Distribution')

sns.histplot(calories_data['Weight'], kde=True, ax=axes[0, 2])
axes[0, 2].axvline(x=weight, color='r', linestyle='--', label='User Input')
axes[0, 2].legend()
axes[0, 2].set_title('Weight Distribution')

sns.histplot(calories_data['Duration'], kde=True, ax=axes[1, 0])
axes[1, 0].axvline(x=duration, color='r', linestyle='--', label='User Input')
axes[1, 0].legend()
axes[1, 0].set_title('Duration Distribution')

sns.histplot(calories_data['Heart_Rate'], kde=True, ax=axes[1, 1])
axes[1, 1].axvline(x=heart_rate, color='r', linestyle='--', label='User Input')
axes[1, 1].legend()
axes[1, 1].set_title('Heart Rate Distribution')

sns.histplot(calories_data['Body_Temp'], kde=True, ax=axes[1, 2])
axes[1, 2].axvline(x=body_temp, color='r', linestyle='--', label='User Input')
axes[1, 2].legend()
axes[1, 2].set_title('Body Temperature Distribution')

# Show plots using st.pyplot()
st.pyplot(fig)

# Model performance on test data
st.subheader('Model Performance')
Y_pred = model.predict(X_test)
mae = metrics.mean_absolute_error(Y_test, Y_pred)
mse = metrics.mean_squared_error(Y_test, Y_pred)
rmse = round(mse ** 0.5, 2)
st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Root Mean Squared Error: {rmse:.2f}')

# Show some of the prediction data
st.subheader('Sample Prediction Data')
st.write(input_data)
