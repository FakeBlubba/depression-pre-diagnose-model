import sys
from pathlib import Path
from process_datasets import add_wordnet_info_to_token
import sentiment_analysis as sa
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import utils as ut
from nltk.stem import SnowballStemmer

def generate_output_score(total_cases, positive_cases, weight):
    output = ((weight * positive_cases) / total_cases) * 10 
    if output >= 1:
        return 1
    return round(output, 2) 


def generate_sentiment_analysis_score(input, weight):
    return round(sa.sentiment_analysis(input) * weight, 2)

def process_input(input_text):
    stemmer = SnowballStemmer('english')
    infos = add_wordnet_info_to_token(input)
    return set([stemmer.stem(token) for index, token in enumerate(word_tokenize(input_text + infos)) if token.lower() not in stopwords.words('english') and token.lower() not in string.punctuation])

def generate_wordnet_score(input, depression_counter_dictionary, not_depressed_counter_dictionary, weight):
    input_set = process_input(input)
    positive_cases = 0
    for word in depression_counter_dictionary.keys():
        if word in input_set:
            positive_cases += depression_counter_dictionary[word]
        
    total = positive_cases    
    for word in not_depressed_counter_dictionary.keys():
        total += not_depressed_counter_dictionary[word]
    
    return generate_output_score(total, positive_cases, weight)

def generate_framenet_score(input, framenet_depression, framenet_not_depression, weight):
    score = 0.1
    return score

def generate_prediction_from_sentence(wordnet_score, framenet_score, sentiment_analysis_score, threshold):
    score = wordnet_score + framenet_score + sentiment_analysis_score
    return 1 if score >= threshold else 0
