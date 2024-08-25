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
import os

import kagglehub

# Download latest version
path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/cmlm-en-large")

print("Path to model files:", path)



class BertDepressionClassifier:
    def __init__(self,
                preprocessor_url = "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3",
                encoder_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-en-large/1",
                learning_rate = 0.016):
        """
        Initialize the BertDepressionClassifier with model URL, tokenizer name

        Args:
            prprocessor_url (str): The URL of the BERT model.
            encoder_url (str): The name of the BERT tokenizer.
        """
        self.preprocessor = hub.KerasLayer(preprocessor_url)
        self.encoder = hub.KerasLayer(encoder_url)
        self.learning_rate = learning_rate

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
        # Input BERT Layers
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        print(input_layer)
        preprocessed_text = self.preprocessor(input_layer)  
        outputs = self.encoder(preprocessed_text)  
        
        # Adding Dense Layers
        x = tf.keras.layers.Dropout(0.2, name='dropout_1')(outputs['pooled_output'])  
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)          
        x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)                         
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)           
        x = tf.keras.layers.Dropout(0.2, name='dropout_3')(x)                         

        # Output Layer
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        return model



    def compile_model(self, model):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)


        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ]

        model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = METRICS)
        return model
    

def main():
    try:
        classifier = BertDepressionClassifier()
        print('CLASSIFICATORE')
        d = classifier.get_data_processed("composite_db.csv")
        print("DATASET")
        X_train, X_test, y_train, y_test = classifier.train_data(d)
        print(X_train[:5])
        

        X_train = tf.convert_to_tensor(X_train, dtype=tf.string)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.string)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

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
    
    except Exception as e:
        print(f"An error occurred: {e}")

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
