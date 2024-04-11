from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import spacy
import nltk
import pickle
import pandas as pd
from spacy.tokens import Token
from sklearn.model_selection import train_test_split

Token.set_extension('lemma', default=None)
nlp = spacy.load('en_core_web_sm')

stpwrd = nltk.corpus.stopwords.words('english')

# lemminf=[]
# for i in range(len(df)):
#   words=[]
#   doc = nlp(df['Reports'].iloc[i])
#   for token in doc:
#     if str(token) not in stpwrd:
#       words.append(token.lemma_)
#   lemminf.append((" ".join(words)))

# vectorizer = TfidfVectorizer()
# vectorizer.fit(lemminf)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model2.pkl', 'rb') as f:
    model = pickle.load(f)

# /print vectorizer output to see the parameters

app = Flask(__name__)

# Load your trained model
# model = joblib.load("final_model_classification.joblib")


@app.route('/classify', methods=['POST'])
def classify_complaint():
    # Get the complaint text from the form data
    complaint_text = request.form.get('complaint')

    words=[]
    doc = nlp(complaint_text)
    
    for token in doc:
        if str(token) not in stpwrd:
            words.append(token.lemma_)

    corpus = (' '.join(words))

    # vectorizer = TfidfVectorizer()
    E = vectorizer.transform([corpus])

    # Pass the complaint text to the model for classification
    prediction = model.predict(E)

    # Return the classification result
    return jsonify({'prediction': prediction[0]})

app.run()