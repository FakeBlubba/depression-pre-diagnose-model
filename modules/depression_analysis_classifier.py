from sklearn.model_selection import cross_val_score
from pathlib import Path
import manage_datasets as md
import process_datasets
import score_generator
import audio_manager
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

class DepressionAnalysisClassifier(BaseEstimator, ClassifierMixin):
    """
    The classifier that predicts depression based on text analysis using different linguistic features.
    
    Attributes:
        percentage_to_maintain (float): The percentage of top words to keep when analyzing frequent words.
        threshold (float): The score threshold above which the classifier predicts depression.
        wn_weight (float): The weight assigned to the WordNet-derived features.
        fn_weight (float): The weight assigned to the FrameNet-derived features.
        sa_weight (float): The weight automatically calculated for sentiment analysis features.
    """
    def __init__(self, percentage_to_maintain = 0.1, threshold = 0.6, wn_weight = 1/2.5 , fn_weight = 1/2.5):
        """
        Initializes the DepressionAnalysisClassifier with specified weights and threshold for classification.

        Args:
            percentage_to_maintain (float): Percentage of frequent words to retain in the analysis.
            threshold (float): Threshold for deciding between depressed and not depressed outcomes.
            wn_weight (float): Weight for the WordNet score contribution.
            fn_weight (float): Weight for the FrameNet score contribution.
        """
        self.percentage_to_maintain = percentage_to_maintain
        self.threshold = threshold
        self.wn_weight = wn_weight
        self.fn_weight = fn_weight
        self.sa_weight = 1 - (wn_weight + fn_weight)
        
    # fit data to be useful following the standard
    def fit(self, X, y):
        """
        Fit the DepressionAnalysisClassifier model to the training data.

        Args:
            X (list of str): The input texts.
            y (list of int): The target labels (1 for depressed, 0 for not depressed).

        Returns:
            self: The instance of the classifier.
        """
        print("starting fit...")
        depressed_data = [[index, X[index], y[index]] for index, target in enumerate(y) if target == 1]
        not_depressed_data = [[index,X[index], y[index]] for index, target in enumerate(y) if target == 0]
        depressed_words = process_datasets.get_infos_from_list_of_sentences(depressed_data)
        not_depressed_words = process_datasets.get_infos_from_list_of_sentences(not_depressed_data)
        
        # WordNet
        self.depressed_word_counter = process_datasets.get_frequent_words(depressed_words, self.percentage_to_maintain)
        self.not_depressed_word_counter = process_datasets.get_frequent_words(not_depressed_words, self.percentage_to_maintain)
        
        # FrameNet
        self.framenet_word_counter = process_datasets.get_framenet_words(depressed_words)
        self.framenet_word_counter = process_datasets.get_framenet_words(not_depressed_words)

        
        print('ending fit...')
        return self

    #   Predict the values
    def predict(self, X):
        print("starting predict...")
        if not isinstance(X, list):
            if isinstance(X, str): 
                X = [X]
        try:
            predictions = []
            for text in X:
                wn_score = score_generator.generate_score(text, self.depressed_word_counter, self.not_depressed_word_counter, self.wn_weight)
                fm_score = score_generator.generate_score(text, None, None, self.fn_weight)
                sa_score = score_generator.generate_sentiment_analysis_score(text, self.sa_weight)
                print(f"wordnet: {wn_score},\t framenet: {fm_score},\t sentiment_analysis: {sa_score}")
                prediction = score_generator.generate_prediction_from_sentence(wn_score, fm_score, sa_score, self.threshold)
                print(prediction)
                predictions.append(prediction)
                print("ending predict.")
                
        except Exception as e:
            print(f"Error in predict: {e}")
            pass
        return np.array(predictions)
    