import numpy as np
import time
import string
import ast
import heapq
from collections import Counter
from nltk.stem import SnowballStemmer

time_then = time.time()

stemmer = SnowballStemmer(language='english')

stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no',
              'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 'there', 'these','they', 'this',
              'to', 'was', 'will', 'with', 'we']

puncts = string.punctuation

key_word_fillers = ['of', 'in', 'on' , 'for']

def key_words(sentence):
    '''Defines word as 'important' if the word comes after: of, in, on or for'''
    key_word = False
    key_words = []
    for idx, word in enumerate(sentence.split()):
        if word in key_word_fillers:
            key_word = True
        if key_word and word not in key_word_fillers:
            key_words.append(word.lower())
    if key_words == []:
        key_words = sentence.lower().split()
    return key_words

def n_grams(query):
    '''Returns list of 2-grams of words in query'''
    query_list = [word for word in query.split() if word not in stop_words]

    query_n_grams = []
    for idx, q in enumerate(query_list):
        if len(query_list[idx:idx+2]) != 1:
            q = ' '.join(query_list[idx:idx+2])
            query_n_grams.append(q)
    return query_n_grams

def giving_weight(sim_dict_words):
    '''Gives appropriate weights to query words'''
    if len(sim_dict_words) > 4:
        avg_sim_score = np.mean(list(sim_dict_words.values()))

        max_keys = sorted(sim_dict_words.items(), key=lambda x: x[1])

        for key, val in dict(max_keys[:2]).items():
            sim_dict_words[key] = val + (avg_sim_score)
    
    max_keys = sorted(sim_dict_words.items(), key=lambda x: x[1])
    for key, val in dict(max_keys[:2]).items():
            sim_dict_words[max(sim_dict_words)] = sim_dict_words.get(min(sim_dict_words))
    
    return sim_dict_words


def similarity_compute(words, nlp):
    '''Returns word similarity to other words in a query'''
    main_word = ''
    dict_words = {}
    words_list = words.split()
    
    for idx, word in enumerate(words_list):
        if word not in stop_words:
            main_word = nlp(word)
            score_list = []

            other_words = words_list[:idx] + words_list[idx + 1:]

            for other_word in other_words:
                score_list.append(main_word.similarity(nlp(other_word)))

            dict_words[str(main_word)] = np.sum(score_list)
    weight_dict_words = giving_weight(dict_words)
    return weight_dict_words

def find_related_words(word, word_vectors, topn=1):
    try:
        similar_words = word_vectors.most_similar(word, topn=topn)
        return [word for word, _ in similar_words]
    except KeyError:
        return []
    

def algorithm_v3(query, word_vectors, nlp, dataset):
    if len(query.split()) == 1:
        min_score = 1
    else:
        min_score = 0.1
    
    important_words = [stemmer.stem(word) for word in key_words(query)]
    query_words = set(word.lower() for word in query.split() if word.lower() not in stop_words) # removing any stop words
    similar_query_words = set(word_s for word in query_words for word_s in find_related_words(word, word_vectors)) # finding related words in query_words
    query_words |= similar_query_words # combining similar_query_words and query_words
    weight_words = similarity_compute(' '.join(query_words), nlp) # giving weight to the words
    
    query_words = set([stemmer.stem(word) for word in query_words])
    
    for key in weight_words.copy().keys():
        weight_words[stemmer.stem(key)] = weight_words[key]
        if stemmer.stem(key) != key:
            del weight_words[key]    
    heap = []
    query_counter = Counter(query_words)
    # going through combined dataset variable
    for idx, val in dataset.items():
        if all(word in val for word in important_words):
            val = ast.literal_eval(val)
            n_gram_words = list(set(n_grams(' '.join(val))) & set(n_grams(' '.join([stemmer.stem(word) for word in query.lower().split(' ')]))))
            n_gram_words_count = 1 if len(n_gram_words) == 0 else len(n_gram_words)
            
            # finding exact same words from dataset and query_words variable
            val_counter = Counter(val)
            same_words = list((query_counter & val_counter))
            
            # determining score by word weights
            score = np.sum([weight_words.get(i)*n_gram_words_count for i in same_words])

            
            if score >= min_score:
                if len(heap) < 10:
                    heapq.heappush(heap, (score, idx))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, idx))
                else:
                    continue
    
    
    # sorts indexes by score values
    top10_idx = [idx for score, idx in heapq.nlargest(10, heap)]
    
    return top10_idx


time_now = time.time()
print(time_now-time_then)