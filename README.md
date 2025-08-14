# SMS Spam Classifier

## Overview
This project is a web-based SMS spam classifier built using Streamlit. It leverages natural language processing (NLP) techniques and a machine learning model to predict whether a given SMS message is spam or not.

## Features
- User-friendly web interface powered by Streamlit
- Real-time SMS spam prediction
- Text preprocessing using NLTK (tokenization, stopword removal, stemming)
- Machine learning model trained to classify messages as 'Spam' or 'Not Spam'

## How It Works
1. **Text Preprocessing**: The input message is cleaned, tokenized, and stemmed using NLTK.
2. **Vectorization**: The processed text is transformed into numerical features using a pre-trained TF-IDF vectorizer (`Vectorizer.pkl`).
3. **Machine Learning Model**: The project uses a supervised learning algorithm for binary classification (spam vs. not spam). The model is trained on a labeled SMS dataset, where each message is marked as either spam or ham (not spam). Typical models for this task include Multinomial Naive Bayes, Logistic Regression, or Support Vector Machines, which are well-suited for text classification problems. The model learns to identify patterns and keywords commonly found in spam messages, such as promotional language, suspicious links, and certain word frequencies.

**Training Process:**
- The dataset is preprocessed (cleaning, tokenization, stemming, stopword removal).
- Features are extracted using TF-IDF vectorization.
- The model is trained and validated to optimize accuracy and minimize false positives/negatives.
- The trained model is saved as `model_spam_ham.pkl` for inference.

**Features Used:**
- Word frequency and presence
- TF-IDF scores
- Message length
- Presence of special characters or links

4. **Prediction**: The vectorized input is passed to the trained classification model (`model_spam_ham.pkl`) to predict spam or not spam.
5. **Result Display**: The prediction result is shown on the web interface.

## File Structure
- `app.py`: Main Streamlit application.
- `Vectorizer.pkl`: Pre-trained TF-IDF vectorizer (binary).
- `model_spam_ham.pkl`: Trained classification model (binary).
- `transform_text.pkl`: (binary, may contain additional preprocessing logic).
- `requirements.txt`: Python dependencies.
- `Procfile`: Deployment configuration for platforms like Heroku.
- `setup.sh`: Shell script for environment setup (used in deployment).
- `.gitignore`: Ignores virtual environment files.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/rana8aaskar/Spam-sms-classifier.git
   cd Spam-sms-classifier
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Deployment
- The project is ready for deployment on platforms like Heroku using the provided `Procfile` and `setup.sh`.

## Live Demo
Access the deployed application here: [SMS Spam Classifier](https://spam-sms-classifier-7hmt.onrender.com)

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Usage
- Enter an SMS message in the text area and click 'Predict' to see if it is spam or not.


## Author
- [rana8aaskar](https://github.com/rana8aaskar)
