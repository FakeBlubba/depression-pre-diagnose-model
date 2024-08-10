
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # For CUDA
# pip3 install torch torchvision torchaudio # For no GPU

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import sys
import string
import os
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
#from official.nlp import optimization
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
import model_performance
import manage_datasets as md
import process_datasets
import score_generator
import audio_manager
import dataset_checker
from depression_analysis_classifier import DepressionAnalysisClassifier

import kagglehub

class BertDepressionClassifier:
    def __init__(self, 
                preprocessor_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', 
                encoder_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-en-large/1"):
        """
        Initialize the BertDepressionClassifier with model URL, tokenizer name

        Args:
            prprocessor_url (str): The URL of the BERT model.
            encoder_url (str): The name of the BERT tokenizer.
        """
        self.preprocessor = hub.KerasLayer(preprocessor_url)
        self.encoder = hub.KerasLayer(encoder_url)

    def get_data_processed(self, database_name):
        texts, labels = md.load_database(database_name)
        df = {'Data': texts, 'Value': labels}
        return df

    def train_data(self, dataframe_balanced):
        X_train, X_test, y_train, y_test = train_test_split(dataframe_balanced['Data'], dataframe_balanced['Value'], stratify = dataframe_balanced['Value'])
        return X_train, X_test, y_train, y_test

    def get_sentences_embedding(self, sentences):
        preprocessed_text = self.preprocessor(sentences)
        return self.encoder(preprocessed_text)['pooled_output']

    def compare_two_sentences(self, first_sentence, second_sentence):
        return cosine_similarity([first_sentence], [second_sentence])

    def create_model(self):

        # Bert Layers
        input_layer = tf.keras.layers.Input(shape = (), dtype = tf.string, name = "text")
        preprocessed_text = self.preprocessor(input_layer)
        outputs = self.encoder(preprocessed_text)

        # Neural Network Layers
        layer = tf.keras.layers.Dropout(0.1, name = 'dropout')(outputs['pooled_output'])    # Regularization to Avoid overfitting
        layer = tf.keras.layers.Dense(1, activation = 'sigmoid', name = "output")(layer)    # Converts output between 0 and 1
        model = tf.keras.Model(inputs = [input_layer], outputs = [layer])

        return model

    def create_RNN_layers(input_layer, outputs, rnn_layers): # [{Dropout_rate: int, Dense_layer_neuron_number: int, activation_type: str}]
        
        for part_layer in rnn_layers:
            layer = tf.keras.layers.Dropout(0.1, name = 'dropout')(outputs['pooled_output'])    # Regularization to Avoid overfitting
        layer = tf.keras.layers.Dense(1, activation = 'sigmoid', name = "output")(layer)    # Converts output between 0 and 1
        return tf.keras.Model(inputs = [input_layer], outputs = [layer])
    
    def compile_model(self, model):
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ]
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = METRICS)
        return model

    def preprocess_text(self, text, use_wordnet=False):
        """
        Preprocesses the given text by tokenizing, removing stopwords, and applying stemming.
        
        Args:
            text (str): The text to preprocess.
            use_wordnet (bool): Whether to use WordNet for additional processing.
        
        Returns:
            str: The processed text.
        """
        tokens = word_tokenize(text.lower())
        
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        
        if use_wordnet:
            synset = process_datasets.get_main_word_synset(text)
            if synset:
                lemma_names = synset.lemma_names()
                definition = synset.definition()
                synset_info = [lemma_names, definition]
                examples = synset.examples()
                if examples:
                    synset_info.append(f"{examples}")
                tokens = [preprocess_text(str(s)) for s in synset_info]
                tokens = list(set([item for sublist in tokens for item in sublist]))


def main():
    classifier = BertDepressionClassifier()
    '''classifier.build_model()
    
    # Preprocess the data
    data_dict = classifier.split_sets()
    dataloaders = classifier.create_dataloaders(data_dict)
    
    # Train the model
    classifier.fit(dataloaders['train'], dataloaders['val'], epochs=3)
    
    # Make predictions
    test_texts = ["I feel sad today", "I'm really happy"]
    predictions = classifier.predict(test_texts)
    print("Predictions:", predictions)'''
    
    d = classifier.get_data_processed("composite_db.csv")
    X_train, X_test, y_train, y_test = classifier.train_data(d)
    model = classifier.create_model()
    model = classifier.compile_model(model)
    model.fit(X_train, y_train, epochs = 3)
    model.evaluate(X_test, y_test)


    y_predicted = model.predict(X_test)
    y_predicted = y_predicted.flatten()
    y_predicted = np.where(y_predicted > 0.5, 1, 0)
    print(y_predicted)
    
    
    # Metrics
    cm = confusion_matrix(y_test, y_predicted)
    print(f"Confusion Matrix\n{cm}\n{classification_report(y_test, y_predicted)}")
main()



'''d = split_sets()
loaded = create_dataloaders(d, batch_size=32)
for batch in loaded["train"]:
    input_ids, attention_masks, labels = batch
    # Ogni frase ha diversi ID, non è detto che corrispondano a ogni parola
    print(f"Input IDs: {input_ids}")    
    # Ogni frase ha diversi attention mask (1), servono a dire se il singolo token deve essere considerato oppure se no (0)
    print(f"Attention Masks: {attention_masks}")  
    # Le etichette del database
    print(f"Labels: {labels}")
    break  '''

