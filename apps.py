from flask import Flask, render_template, request
from sengine_algorithm import algorithm_v3
from auto_fill_algorithm import auto_fill
import spacy
from gensim.models import KeyedVectors
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)

df_at = pd.read_csv('csv_files/df.csv')
df_at = df_at.iloc[:, 0]
df = pd.read_csv('csv_files/df_modified.csv')

word_vectors = KeyedVectors.load_word2vec_format('models/glove.6B.300d.txt', binary=False, no_header=True)

nlp = spacy.load('en_core_web_md')


genai.configure(api_key='YOUR_GENAI_API_KEY')

model = genai.GenerativeModel('gemini-pro')

def llm_search_corrector(query):
    response = model.generate_content(f'You are a Query Spell Fixer. Given a query, your task is to identify and correct any misspelled words. Your goal is to provide the correct spelling for each word, only return correctly spelled query: {query}')
    return response.text


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', title="Home")

@app.route('/paper/<string:idx>', methods=['GET', 'POST'])
def research_paper(idx):
    result = df.iloc[int(idx)]
    return render_template('paper_results.html', data=result)

@app.route('/search', methods=['GET','POST'])
def search():
    user_query = request.args.get('query')
    if user_query:
        filled_query = auto_fill(user_query)
        if filled_query:
            user_query = ' '.join(user_query.split(' ')[:-1]) + ' ' + filled_query[0]
        print(user_query)
        user_query = llm_search_corrector(user_query)
        print(type(user_query), user_query)
        result_idxs = algorithm_v3(user_query, word_vectors, nlp, df_at)
        results = df.iloc[result_idxs]
        print(result_idxs)
        user_query_list = user_query.split()
        return render_template('results.html', results=results, user_query=user_query_list, search_query=user_query)
    else:
        return "User didn't input anything."


if __name__ == '__main__':
    app.run(debug=True)