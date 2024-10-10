
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # For CUDA
# pip3 install torch torchvision torchaudio # For no GPU

import numpy as np
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

from pathlib import Path
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
import manage_datasets as md



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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
        """
        Loads and processes a dataset from the specified database name. The dataset consists of textual data and corresponding labels.

        Args:
            database_name (str): The name of the database from which to load data.
        
        Returns:
            dict: A dictionary with 'Data' containing the texts and 'Value' containing the labels.
        """
        texts, labels = md.load_database(database_name)
        df = {'Data': texts, 'Value': labels}
        return df

    def train_data(self, dataframe_balanced):
        """
        Splits the provided balanced dataframe into training and test sets, stratifying by the label values to ensure balanced classes.

        Args:
            dataframe_balanced (pd.DataFrame): The dataframe containing both textual data and their associated labels.
        
        Returns:
            tuple: Four arrays corresponding to the training data (X_train), test data (X_test), training labels (y_train), and test labels (y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(dataframe_balanced['Data'], dataframe_balanced['Value'], stratify = dataframe_balanced['Value'])
        return X_train, X_test, y_train, y_test


    def create_model(self):
        """
        Constructs a BERT-based deep learning model with dropout regularization and dense layers. The model uses BERT's pooled output as the input for the dense layers, which aim to classify text data into depression-related or non-depression-related categories.

        Returns:
            tf.keras.Model: A compiled TensorFlow model ready for training and evaluation.
        """
        # Input BERT Layers
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        preprocessed_text = self.preprocessor(input_layer)  
        outputs = self.encoder(preprocessed_text)  

        # Adding Dense Layers
        # Parameters: 
        #   - Units: number of neurons
        #       - Higher Number: capture more specs
        #       - Lower Number: Prevent Overfitting
        #   - Activation: type of function desidered
        #
        # Higher layer number: -
        x = tf.keras.layers.Dropout(0.2, name='dropout_1')(outputs['pooled_output'])  # Regularization: 0.x is the percentage of neurons to deactivate to prevent overfitting
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)          
        x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)                         
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)           
        x = tf.keras.layers.Dropout(0.2, name='dropout_3')(x)                         

        # Output Layer
        output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        return model


    def compile_model(self, model):
        """
        Compiles the given model using the Adam optimizer and sets binary cross-entropy as the loss function. It also defines the metrics to be monitored during training, which include accuracy, precision, and recall.

        Args:
            model (tf.keras.Model): The model to compile.
        
        Returns:
            tf.keras.Model: The compiled TensorFlow model.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)

        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ]

        model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = METRICS)
        return model
    

