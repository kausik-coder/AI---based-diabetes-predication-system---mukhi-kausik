import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

st.title("AI-Based Diabetes Prediction System")

# Inputs
highbp = st.selectbox("High Blood Pressure", [0, 1])
highchol = st.selectbox("High Cholesterol", [0, 1])
cholcheck = st.selectbox("Cholesterol Check", [0, 1])

bmi = st.number_input(
    "Enter BMI",
    min_value=10.0,
    max_value=60.0,
    key="bmi"
)

smoker = st.selectbox("Smoker", [0, 1])

stroke = st.selectbox("Stroke", [0, 1])

heart = st.selectbox("Heart Disease or Attack", [0, 1])

physactivity = st.selectbox("Physical Activity", [0, 1])

fruits = st.selectbox("Fruits", [0, 1])

veggies = st.selectbox("Veggies", [0, 1])

hvyalcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])

anyhealthcare = st.selectbox("Any Healthcare", [0, 1])

nodocbccost = st.selectbox("No Doctor Because of Cost", [0, 1])

genhlth = st.selectbox("General Health (1-5)", [1, 2, 3, 4, 5])

menthlth = st.number_input(
    "Mental Health Days",
    min_value=0,
    max_value=30,
    key="menthlth"
)

physhlth = st.number_input(
    "Physical Health Days",
    min_value=0,
    max_value=30,
    key="physhlth"
)

diffwalk = st.selectbox("Difficulty Walking", [0, 1])

sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])

age = st.number_input(
    "Age Category (1-13)",
    min_value=1,
    max_value=13,
    key="age"
)

education = st.number_input(
    "Education Level (1-6)",
    min_value=1,
    max_value=6,
    key="education"
)

income = st.number_input(
    "Income Level (1-8)",
    min_value=1,
    max_value=8,
    key="income"
)

# Prediction
if st.button("Predict"):

    data = pd.DataFrame([[
        highbp,
        highchol,
        cholcheck,
        bmi,
        smoker,
        stroke,
        heart,
        physactivity,
        fruits,
        veggies,
        hvyalcohol,
        anyhealthcare,
        nodocbccost,
        genhlth,
        menthlth,
        physhlth,
        diffwalk,
        sex,
        age,
        education,
        income
    ]], columns=[
        'HighBP',
        'HighChol',
        'CholCheck',
        'BMI',
        'Smoker',
        'Stroke',
        'HeartDiseaseorAttack',
        'PhysActivity',
        'Fruits',
        'Veggies',
        'HvyAlcoholConsump',
        'AnyHealthcare',
        'NoDocbcCost',
        'GenHlth',
        'MentHlth',
        'PhysHlth',
        'DiffWalk',
        'Sex',
        'Age',
        'Education',
        'Income'
    ])

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1] * 100

    if prediction[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Non-Diabetic")

    st.metric("Diabetes Risk", f"{probability:.2f}%")