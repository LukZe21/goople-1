import Levenshtein
import json

total_words = 100282746

with open('json_files/unique_words.json', 'r') as f:
    words_lst = json.load(f)

with open('json_files/words_count.json', 'r') as f:
    words_count = json.load(f)


def correction(incorrect_word):
    possible_words = {}
    for word in words_lst:
        leven_score = Levenshtein.distance(word, incorrect_word)
        if leven_score <= 4:
            possible_words[word] = leven_score
    try:
        return freq_words(possible_words)

    except:
        return incorrect_word


def freq_words(possible_words):
    freq_dict = {}
    for word, val in zip(possible_words.keys(), possible_words.values()):
        total_occurance = words_count.get(word, 0)
        freq = (total_occurance/total_words)/val
        freq_dict[word] = freq
    return max(freq_dict, key=freq_dict.get)


def check_words(query):
    query_lst = query.lower().split(' ')
    corrected_words = [correction(word) if word not in words_lst else word for word in query_lst]
    return ' '.join(corrected_words)