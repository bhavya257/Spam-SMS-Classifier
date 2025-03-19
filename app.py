import streamlit as st
import joblib
import re
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Clean text function
def clean_text(text):
    text = text.lower()  # Lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [ps.stem(word) for word in tokens]  # Stemming
    return ' '.join(tokens)  # Join tokens into a single string

# Load the pickled vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Streamlit app title
st.title("Spam SMS Classifier")

# Text input from user
user_input = st.text_area("Enter an SMS message:")

# Button to trigger prediction
if st.button("Classify"):
    if user_input:
        # Preprocess the input using clean_text
        processed_input = clean_text(user_input)

        # Vectorize the processed input
        vectorized_input = vectorizer.transform([processed_input]).toarray()

        # Predict using the model
        prediction = model.predict(vectorized_input)[0]

        # Display the result
        result = "Spam" if prediction == 1 else "Not Spam"
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter a message to classify.")