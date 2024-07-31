import sys
from pathlib import Path
from process_datasets import add_wordnet_info_to_token
import sentiment_analysis as sa
import process_datasets
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
    output = ((weight * positive_cases) / total_cases)
    if output >= 1:
        return 1
    return round(output, 2) 

def generate_wn_output_score(input_text, threshold, max_penalty, tree, max_levels):
    """
    Generate a WordNet-based output score for the given input text.

    Args:
        input_text (str): The input text to analyze.
        threshold (float): Threshold for similarity to consider a synset relevant.
        max_penalty (float): Maximum penalty to apply if the synset is not found.
        tree (list): The list of synsets to traverse.
        max_levels (int): Maximum number of levels to explore.

    Returns:
        float: Calculated WordNet-based score.
    """
    main_synset = process_datasets.get_main_word_synset(input_text)
    if main_synset:
        data = process_datasets.find_synset_level(tree, main_synset.name(), threshold, max_levels)
        level = data[0]
        similarity = data[1]

        if level == -1:
            max_similarity = process_datasets.synset_similarity("dog.n.01", "dog.n.01")
            return round(similarity / max_similarity - max_penalty, 2)

        return round(1 - (level * (max_penalty / max_levels)), 2)
    print(f"{input_text} - NOT FOUND MAIN_SYNSET")
    return -1
    

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
    negative_cases = 0

    for word in depression_counter_dictionary:
        if word in input_set:
            dynamic_weight = depression_counter_dictionary[word] / sum(depression_counter_dictionary.values())
            positive_cases += depression_counter_dictionary[word] * dynamic_weight

    for word in not_depressed_counter_dictionary:
        if word in input_set:
            dynamic_weight = not_depressed_counter_dictionary[word] / sum(not_depressed_counter_dictionary.values())
            negative_cases += not_depressed_counter_dictionary[word] * dynamic_weight

    total_cases = positive_cases + negative_cases

    return generate_output_score(total_cases, positive_cases, weight)

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
