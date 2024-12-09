import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import csv


st.markdown("# Welcome to the LinkedIn User Prediction App!")
st.markdown("### Please enter the following information for a prediction: ")


marriage_status_mapping = {
    "Married": 1,
    "Living with a partner": 2,
    "Divorced": 3,
    "Separated": 4,
    "Widowed": 5,
    "Never been married": 6,
    "Don't Know": 8,
    "Refused": 9,
}

# Get user input
marriage_status = st.selectbox(
    "Current Marriage Status:",
    options=list(marriage_status_mapping.keys())  # Display the keys
)

# Convert the selected option to its corresponding value
marriage = marriage_status_mapping[marriage_status]

gender_mapping = {
    "Male":1,
    "Female":2,
    "Other": 3,
    "Don't Know": 98,
    "Refused":99
    
}
# Get user input
gender = st.selectbox(
    "Gender:",
    options=list(gender_mapping.keys())  # Display the keys
)

# Convert the selected option to its corresponding value
gen = gender_mapping[gender]

parent_mapping = {
    "Yes":1,
    "No":2,
    "Don't know ": 8,
    "Refused": 9,
    
}
# Get user input
parent = st.selectbox(
    "Are you a parent of a child under 18 living in your home?",
    options=list(parent_mapping.keys())  # Display the keys
)

# Convert the selected option to its corresponding value
par = parent_mapping[parent]

education_mapping = {
    "Less than high school":1,
    "High school incomplete":2,
    "Some college, no degree": 3,
    "Two-year associate degree": 4,
    "Four-year bachelor's degree": 5,
    "Some postgraduate":7,
    "Postgraduate or professional degree": 8,
    "Don't know":  98,
    "Refused":99,
}
# Get user input
education = st.selectbox(
    "Education Level:",
    options=list(education_mapping.keys())  # Display the keys
)

# Convert the selected option to its corresponding value
edu = education_mapping[education]

income_mapping = {
     "Less than $10,000": 1,
    "10 to under $20,000":2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000":6,
    "100 to under $150,000":7,
    "150,000 or more?":8,
    "Don't know": 98,
    "Refused":99
}
income = st.selectbox(
    "Household Income:",
    options=list(income_mapping.keys())  # Display the keys
)

# Convert the selected option to its corresponding value
income_h =income_mapping[income]


age1 = st.number_input("Age:", min_value=0, max_value=98,step=1)



#st.markdown(
   # f"""
    ### Submission Summary
    #- **Household Income**: {income_h}
   # - **Education**: {edu}
   # - **Parent Status**: {par}
   # - **Marriage Status**: {marriage}
   # - **Gender**: {gen}
   # - **Age**: {age1}
   # """,
#    unsafe_allow_html=True,
#)



s=pd.read_csv('social_media_usage.csv')

# Data cleaning function
def clean_sm(x):
        x=np.where(x == 1, 1, 0)
        return x

# Clean and process data
columns_extracting = ['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']
ss = s.loc[:, columns_extracting]

ss = pd.DataFrame({
        'sm_li': clean_sm(ss['web1h']),                             
        'income': np.where(ss['income'] > 9, np.nan, ss['income']), 
        'education': np.where(ss['educ2'] > 8, np.nan, ss['educ2']),
        'parent': np.where(ss['par'] == 1, 1, 0),                   
        'married': np.where(ss['marital'] == 1, 1, 0),              
        'female': np.where(ss['gender'] == 2, 1, 0),                
        'age': np.where(ss['age'] > 98, np.nan, ss['age'])          
    })

ss=ss.dropna()

y=ss["sm_li"]
X=ss[['income','education','parent','married','female','age']]

X_train, X_test, y_train,y_test=train_test_split(X.values,
                                                 y,
                                                 stratify=y,
                                                 test_size=0.2,
                                                 random_state=987)

lr=LogisticRegression()
lr.fit(X_train,y_train)

person=[income_h, edu, par, marriage,gen,age1]
predicted_class=lr.predict([person])
probs=lr.predict_proba([person])
predicted_message = "LinkedIn User" if predicted_class[0] == 1 else "Non-LinkedIn User"
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"#### Predicted class: <u>{predicted_message}</u>", unsafe_allow_html=True)
st.markdown(f"#### Probability that this person is a LinkedIn user: <u>{probs[0][1]*100:.2f}%</u>", unsafe_allow_html=True)