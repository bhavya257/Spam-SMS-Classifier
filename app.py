import streamlit as st
import joblib
import re
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time

# Download NLTK data
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

# Set page title and favicon
st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="favicon.ico",
)

# Full-page loading screen HTML/CSS with gradient spinner
loading_html = """
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #0E1117; z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div style="color: white; font-size: 24px; text-align: center;">
        <span style="display: inline-block; width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(90deg, rgb(255, 75, 75), rgb(255, 253, 128)); animation: spin 1s linear infinite; position: relative; margin-bottom: 10px;">
            <span style="position: absolute; top: 4px; left: 4px; width: 32px; height: 32px; background: #0E1117; border-radius: 50%;"></span>
        </span>
        <br>Loading Spam SMS Classifier...
    </div>
</div>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

# Show loading screen only on first load
if 'loaded' not in st.session_state:
    st.markdown(loading_html, unsafe_allow_html=True)
    time.sleep(1)  # Simulate loading time
    vectorizer = joblib.load('vectorizer.pkl')  # Load the pickled vectorizer
    model = joblib.load('model.pkl')  # Load the pickled model
    st.session_state['loaded'] = True
    st.session_state['vectorizer'] = vectorizer
    st.session_state['model'] = model
    st.rerun()

# Access loaded vectorizer and model
vectorizer = st.session_state['vectorizer']
model = st.session_state['model']

# Streamlit app title
st.title("Spam SMS Classifier")

# Text input from user
user_input = st.text_area("Enter an SMS message:")

# Custom spinner for classification
spinner_html = """
<div style="display: flex; justify-content: center; align-items: center; padding: 10px;">
    <div style="width: 30px; height: 30px; border: 4px solid #ddd; border-top: 4px solid #ff5733; border-radius: 50%; animation: spin 1s linear infinite;"></div>
    <span style="margin-left: 10px; font-size: 18px; color: #ff5733;">Classifying...</span>
</div>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

# Footer HTML/CSS with gradient link
footer_html = """
<div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 10px 0; text-align: center; z-index: 1000;">
    <p style="margin: 0; font-size: 14px; color: white;">
        Discover more of my work at 
        <a href="https://bhavyajain.in/" target="_blank" style="background: linear-gradient(90deg, rgb(255, 75, 75), rgb(255, 253, 128)); -webkit-background-clip: text; color: transparent; text-decoration: none;">bhavyajain.in</a>.
    </p>
</div>
"""

# Display the footer
st.markdown(footer_html, unsafe_allow_html=True)

# Placeholder for prediction output
prediction_placeholder = st.empty()

# Button to trigger prediction
if st.button("Classify"):
    if user_input:
        with prediction_placeholder.container():
            st.markdown(spinner_html, unsafe_allow_html=True)
            processed_input = clean_text(user_input)  # Preprocess the input using clean_text
            vectorized_input = vectorizer.transform([processed_input]).toarray()  # Vectorize the processed input
            prediction = model.predict(vectorized_input)[0]  # Predict using the model
            time.sleep(0.5)  # Artificial delay
        # Clear placeholder and show new result
        prediction_placeholder.empty()
        with prediction_placeholder.container():
            result = "Spam" if prediction == 1 else "Not Spam"
            st.write(f"Prediction: **{result}**")
    else:
        prediction_placeholder.empty()
        with prediction_placeholder.container():
            st.write("Please enter a message to classify.")