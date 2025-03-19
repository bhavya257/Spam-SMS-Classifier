# Spam SMS Classifier

This is a simple web application built with [Streamlit](https://streamlit.io/) that classifies SMS messages as **Spam** or **Not Spam**. It uses a machine learning model trained on SMS data, with preprocessing steps including tokenization, stop word removal, and stemming. The app takes a user-entered message, processes it, and predicts whether it’s spam using a pre-trained model.

## Features
- Input an SMS message via a text area.
- Click the "Classify" button to get a prediction.
- Displays the result as "Spam" or "Not Spam".

## Repository Structure
```
spam-sms-classifier/
├── app.py            # Main Streamlit app file
├── requirements.txt  # Python dependencies
├── .gitignore        # Files and directories to ignore in Git
├── vectorizer.pkl    # Pre-trained TF-IDF vectorizer
└── model.pkl         # Pre-trained machine learning model
```

## How It Works
1. The app loads a pre-trained TF-IDF vectorizer (`vectorizer.pkl`) and model (`model.pkl`) using `joblib`.
2. User inputs an SMS message in the text area.
3. The message is preprocessed using a `clean_text` function (lowercasing, removing special characters, tokenizing, removing stop words, and stemming).
4. The processed text is vectorized and passed to the model for prediction.
5. The result is displayed on the screen.

## Model Training
The model and vectorizer were trained using a Kaggle notebook. You can find the full training code here:  
[**Kaggle Notebook: Spam SMS Classification Training**](https://www.kaggle.com/code/bhavyajain21bci0308/spam-sms-classifier-model)

## Prerequisites
- Python 3.8+
- Git

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/bhavya257/Spam-SMS-Classifier.git
   cd spam-sms-classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally
1. Ensure all files (`app.py`, `vectorizer.pkl`, `model.pkl`) are in the same directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501` to use the app.

## Deployment
This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). To deploy your own version:
1. Push this repo to GitHub.
2. Sign in to Streamlit Community Cloud with your GitHub account.
3. Create a new app, select your repo, and set `app.py` as the main file.
4. Deploy and access the live app via the provided URL.

## Dependencies
Listed in `requirements.txt`:
- `streamlit`
- `joblib`
- `scikit-learn`
- `nltk`
- `pandas`

## License
This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments
- Built as part of a spam SMS classification project.
- Thanks to the open-source communities.