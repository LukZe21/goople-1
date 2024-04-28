from nltk.corpus import stopwords
import json
from gensim.models import Word2Vec
from word_corrections import check_words


stop_words = stopwords.words('english')

model1 = Word2Vec.load("models/word2vec_model")

with open('json_files/unique_words.json', "r") as json_file:
    unique_lst = json.load(json_file)


def related_words(query_word):
    related_words = {}
    for word in unique_lst:
        try:
            score = model1.wv.similarity(query_word, word) # finds similarity score between query and word in a word list.
            if score >= .5 and score < .99:
                related_words[word] = score
        except:
            continue
    return dict(sorted(related_words.items(), key=lambda item: item[1])) #returns words by ascending order


def auto_fill(prev_words):
    words = prev_words.split(' ')
    prev_words = words[:-1] # all words except last one
    prev_words = check_words(' '.join(prev_words)) # check words mispellings, if so fixes them.
    prev_words = prev_words.split(' ')
    fill_word = words[-1] # word we want to predict.
    possible_outcomes = {}
    if len(words) >= 2:
        for idx, prev_word in enumerate(prev_words):
            if prev_word not in stop_words:
                lst = related_words(prev_word) # finds related words to prev_word 
                for word in lst:
                    # if related words list starts with fill_word then it will give weights, accordingly.
                    if word.startswith(fill_word):
                        try:
                            if prev_words[idx+2]:
                                possible_outcomes[word] = 0.01
                        except:
                            possible_outcomes[word] = 0.02
        # if there is no possible_outcomes it will go through unique words list and will try to find the words that starts with fill_word variable. (gives equal weight to all of the words)
        if possible_outcomes == {}: 
            for word in unique_lst:
                if word.startswith(fill_word):
                    possible_outcomes[word] = 0.5
    return sorted(possible_outcomes, key=possible_outcomes.get, reverse=True)[:10]