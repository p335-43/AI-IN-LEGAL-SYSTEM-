from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

df = pd.read_csv('/Users/pranjalmishra/Desktop/python/ipc_sections.csv', encoding='unicode_escape')

def preprocess_text(text):
    doc = nlp(text.lower())
    filtered_tokens = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop]
    return ' '.join(filtered_tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_question = request.form['text']
    processed_question = preprocess_text(user_question)
    preprocessed_descriptions = [preprocess_text(desc) for desc in df['Description']]
    vectorizer = TfidfVectorizer()
    description_vectors = vectorizer.fit_transform(preprocessed_descriptions)
    question_vector = vectorizer.transform([processed_question])
    similarities = cosine_similarity(question_vector, description_vectors)
    similarity_threshold = 0.1
    similar_indices = np.where(similarities[0] >= similarity_threshold)[0]
    recommended_ipcs = df.loc[similar_indices, ['Section', 'Punishment']]

    return render_template('results.html', question=user_question, recommendations=recommended_ipcs.to_html())

if __name__ == '__main__':
    app.run(port=5000)