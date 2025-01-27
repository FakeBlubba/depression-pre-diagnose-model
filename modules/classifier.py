
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
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import precision_score, recall_score, f1_score

from pathlib import Path
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
import manage_datasets as md
import reports
import kagglehub

'''
# Download latest version
path = kagglehub.model_download("tensorflow/bert/tensorFlow2/en-uncased-preprocess")
path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/cmlm-en-large")
print("Path to model files:", path)'''
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

class BertDepressionClassifier:
    def __init__(self,
                preprocessor_url = "C:\\Users\\Federico\\.cache\\kagglehub\\models\\tensorflow\\bert\\tensorFlow2\\en-uncased-preprocess\\3",
                encoder_url = "C:\\Users\\Federico\\.cache\\kagglehub\\models\\google\\universal-sentence-encoder\\tensorFlow2\\cmlm-en-large\\1",
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
    
    def get_formatted_data(self, train_set_path, test_set_path):
        X_train, X_test, y_train, y_test = md.get_train_and_test_set("binary_dataset.csv")
        return {'Data': X_train + y_train, 'Value': X_test + y_test}

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


    def leave_one_out_cv(self, X, y, epochs = 1, batch_size = 16):
        """
        Implement Leave-One-Out Cross-Validation (LOOCV).
        Train and validate the model for each sample in the dataset, leaving one sample out for testing.
        """
        cv = LeaveOneOut()
        y_pred, y_true = [], []
        
        
        for train_idx, test_idx in cv.split(X):
            X_train_fold, X_test_fold = np.array(X)[train_idx], np.array(X)[test_idx]
            y_train_fold, y_test_fold = np.array(y)[train_idx], np.array(y)[test_idx]
            
            model = self.create_model()
            model = self.compile_model(model)
            model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size = batch_size)
        
            y_predicted = model.predict(X_test_fold)
            y_pred.append(np.where(y_predicted > 0.5, 1, 0)[0])
            y_true.append(y_test_fold[0])


        accuracy = np.mean(np.array(y_true) == np.array(y_pred))  
        precision = precision_score(y_true, y_pred, zero_division=0) 
        recall = recall_score(y_true, y_pred, zero_division=0)  
        f1 = f1_score(y_true, y_pred, zero_division=0)  

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def simple_cross_validation(self, X, y, n_splits=10):
        """
        Run cross-validation

        Args:
            X (array-like): Input Data
            y (array-like): Labels.
            n_splits (int): Number of Splits.

        Returns:
            dict: Metriche medie (accuracy, precision, recall).
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        precisions = []
        recalls = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            model = self.create_model()
            model = self.compile_model(model)
            model.fit(X_train, y_train, epochs=3, verbose=0)

            y_pred = model.predict(X_test).flatten()
            y_pred = np.where(y_pred > 0.5, 1, 0)

            accuracy = np.mean(y_pred == y_test)
            precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
            recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)

        return {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
        }