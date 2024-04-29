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
    """
    Calculate a weighted output score based on the provided case data and weight.

    Args:
        total_cases (int): Total number of cases considered.
        positive_cases (int): Number of positive cases among the total.
        weight (float): Weighting factor to adjust the influence of positive cases.

    Returns:
        float: Normalized score adjusted by weight, scaled to a maximum of 1.
    """
    output = ((weight * positive_cases) / total_cases) * 10 
    if output >= 1:
        return 1
    return round(output, 2) 


def generate_sentiment_analysis_score(input, weight):
    """
    Generates a weighted sentiment analysis score for the given input text.

    Args:
        input (str): The input text to analyze.
        weight (float): Weight factor to apply to the sentiment score.

    Returns:
        float: The sentiment score, weighted and rounded to two decimal places.
    """
    return round(sa.sentiment_analysis(input) * weight, 2)

def process_input(input_text):
    """
    Processes the input text by adding WordNet information, tokenizing, stemming, and removing stopwords.

    Args:
        input_text (str): Text to be processed.

    Returns:
        set: A set of stemmed tokens from the processed input text.
    """
    stemmer = SnowballStemmer('english')
    infos = add_wordnet_info_to_token(input)
    return set([stemmer.stem(token) for index, token in enumerate(word_tokenize(input_text + infos)) if token.lower() not in stopwords.words('english') and token.lower() not in string.punctuation])

def generate_score(input, depression_counter_dictionary, not_depressed_counter_dictionary, weight):
    """
    Generates a score by analyzing the presence of certain words in the input text, using specified word frequency dictionaries.

    Args:
        input (str): The input text to analyze.
        depression_counter_dictionary (dict): Dictionary of word frequencies related to positive cases.
        not_depressed_counter_dictionary (dict): Dictionary of word frequencies not related to negative cases.
        weight (float): Weight factor.

    Returns:
        float: Calculated score based on the input and dictionaries.
    """
    input_set = process_input(input)
    positive_cases = 0
    for word in depression_counter_dictionary.keys():
        if word in input_set:
            positive_cases += depression_counter_dictionary[word]
        
    total = positive_cases    
    for word in not_depressed_counter_dictionary.keys():
        total += not_depressed_counter_dictionary[word]
    
    return generate_output_score(total, positive_cases, weight)

def generate_prediction_from_sentence(wordnet_score, framenet_score, sentiment_analysis_score, threshold):
    """
    Generates a binary prediction from scores based on WordNet, FrameNet, and sentiment analysis.

    Args:
        wordnet_score (float): Score derived from WordNet analysis.
        framenet_score (float): Score derived from FrameNet analysis.
        sentiment_analysis_score (float): Score derived from sentiment analysis.
        threshold (float): Threshold value to determine the prediction outcome.

    Returns:
        int: 1 if the combined score is above the threshold, otherwise 0.
    """
    score = wordnet_score + framenet_score + sentiment_analysis_score
    return 1 if score >= threshold else 0
