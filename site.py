from flask import Flask,request,render_template,url_for,jsonify
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
import streamlit as st
# from flask_cors import CORS

 # type: ignore
# Preprocessing function
stop_words = set(stopwords.words('english'))  # Load stopwords

def datapreprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text_token = word_tokenize(text)  # Tokenize into words
    filtered_text = [w for w in text_token if w not in stop_words]  # Remove stopwords
    return filtered_text

# Function to create a vector for a single essay
def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")  # Initialize zero vector
    noOfWords = 0.  # Counter for valid words

    index2word_set = set(model.index_to_key)  # Use `model.index_to_key` directly
    for word in words:
        if word in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model[word])  # Access word vector

    if noOfWords > 0:
        vec = np.divide(vec, noOfWords)  # Average the vectors

    return vec

# Function to create vectors for all essays
def getVecs(essays, model, num_features):
    essay_vecs = np.zeros((len(essays), num_features), dtype="float32")  # Initialize a matrix
    for c, essay in enumerate(essays):
        essay_vecs[c] = makeVec(essay, model, num_features)  # Convert each essay into a vector
    return essay_vecs  

# Function to predict essay scores
def convertToVec(text):
    try:
        if len(text) > 20:  # Ensure the essay has sufficient content
            num_features = 300
            # Load Word2Vec model
            # st.write("Loading Word2Vec model...")
            model = KeyedVectors.load_word2vec_format("word2vec_format.bin", binary=True)
            # st.write("Word2Vec model loaded successfully.")
            
            # Preprocess the input essay
            # st.write("Preprocessing text...")
            clean_test_essays = [datapreprocess(text)]
            # st.write("Cleaned text:", clean_test_essays)
            
            # Generate vectors
            testDataVecs = getVecs(clean_test_essays, model, num_features)
            testDataVecs = np.array(testDataVecs)
            testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))  # Reshape for LSTM

            # Load pre-trained LSTM model
            # st.write("Loading LSTM model...")
            lstm_model = load_model("final_lstm.h5")
            # st.write("LSTM model loaded successfully.")
            
            # Predict the score
            st.write("Generating prediction...")
            preds = lstm_model.predict(testDataVecs)
            st.write("Prediction:", preds)
            return str(round(preds[0][0]))
        else:
            return "0"  # Return 0 for very short essays
    except Exception as e:
        return f"Error: {str(e)}"  # Handle errors gracefully

# Streamlit App
def main():
    st.title("Automated Essay Scoring")
    st.write("Enter your essay in the text box below to predict the score.")

    # Text area for essay input
    essay_input = st.text_area("Essay Input", placeholder="Type or paste your essay here...")

    if st.button("Score my Essay!"):
        if len(essay_input.strip()) == 0:
            st.error("Please enter some text before scoring.")
        else:
            st.write("Processing...")
            score = convertToVec(essay_input)
            if "Error" in score:
                st.error(f"An error occurred: {score}")
            else:
                st.success(f"Predicted Score: {score}")

if __name__ == "__main__":
    main()

    