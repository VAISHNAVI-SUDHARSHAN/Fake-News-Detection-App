#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install streamlit nltk scikit-learn')



# Import necessary libraries
import streamlit as st
import pandas as pd
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Load the trained model (make sure you have 'model.pkl' file in the same directory)
# If you haven't saved the model, refer to the model training code in the previous steps
# and save it using pickle.
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to clean the input text (same as the one used during training)
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

# Streamlit web app interface
st.title("Fake News Classifier")

# Create a text input for the user to enter news content
news_input = st.text_area("Enter the news text you want to classify:")

# When the user clicks the 'Predict' button, perform prediction
if st.button("Predict"):
    if news_input.strip() == "":
        st.write("Please enter some news text to get a prediction.")
    else:
        # Clean the input text
        news_cleaned = clean_text(news_input)
        
        # Make prediction using the loaded model
        prediction = model.predict([news_cleaned])
        
        # Display the result
        result = 'Real News' if prediction[0] == 1 else 'Fake News'
        st.write(f"The entered news is classified as: **{result}**")


# In[ ]:




