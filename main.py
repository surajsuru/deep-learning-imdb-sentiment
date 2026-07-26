# Import libraries and load the model

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## load the IMDB dataset word index

word_index = imdb.get_word_index()
reverse_word_index = { value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

# Step2: Helper Functions
# Functions to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to preprocess user input
# The model was trained with reviews padded to length 500
# so we must use the same length here

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## prediction

def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


## streamlit app

import streamlit as st

st.title("IMDB Movie Review Sentiment analysis")
st.write("Enter a Movie review to classify it as positive or negative")

#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input  = preprocess_text(user_input)

    ##make prediction 
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction}')

else:
    st.write('please enter a movie review.')
