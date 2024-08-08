
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
                encoder_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'):
        """
        Initialize the BertDepressionClassifier with model URL, tokenizer name, maximum token length, and batch size.
        
        Args:
            model_url (str): The URL of the BERT model.
            tokenizer_name (str): The name of the BERT tokenizer.
            max_length (int): The maximum length of tokenized sequences.
            batch_size (int): The batch size for DataLoader.
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

    def compile_model(self, model):
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ]
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = METRICS)
        return model
        
    
    
    
    
    
    
    def build_model(self, print = False):
        """
        Builds the BERT-based classification model.
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer("https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-preprocess/3")
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer("https://www.kaggle.com/models/tensorflow/bert/TensorFlow2/bert-en-uncased-l-8-h-768-a-12/2", trainable=True)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]
        
        # Add a dense layer for classification
        x = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)
        
        model = tf.keras.Model(text_input, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
            loss='binary_crossentropy', 
            metrics=['accuracy'])
        
        self.model = model  
        
        if print:
            self.print_pooled_output(text_input, pooled_output)

        
    def print_pooled_output(self, text_input, pooled_output, text = "I am really depressed"):
        embedding_model = tf.keras.Model(text_input, pooled_output)
        sentences = tf.constant([text])
        print(embedding_model(sentences))
        
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

        return " ".join(tokens)

    def preprocess_data(self, file_name, use_wordnet=False):
        """
        Processes the dataset by reading from a CSV file and applying preprocessing to each row.
        
        Args:
            file_name (str): The name of the CSV file to process.
            use_wordnet (bool): Whether to use WordNet for additional processing.
        
        Returns:
            list: A list of processed tokens for each row.
        """
        path = os.path.join(md.get_DB_path(), file_name)
        df = pd.read_csv(path)
        
        sentences = df['Data'].tolist()
        
        processed_sentences = [self.preprocess_text(sentence, use_wordnet) for sentence in sentences]
        
        return processed_sentences

    def encode_data(self, texts, labels):
        """
        Encodes the texts and labels for BERT input.
        
        Args:
            texts (list): A list of text samples to encode.
            labels (list): A list of labels corresponding to the text samples.
        
        Returns:
            TensorDataset: A dataset containing input IDs, attention masks, and labels.
        """
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length)
        input_ids = torch.tensor(encodings['input_ids'])
        attention_masks = torch.tensor(encodings['attention_mask'])
        labels = torch.tensor(labels)
        return TensorDataset(input_ids, attention_masks, labels)

    def split_sets(self, db_name="composite_db.csv", test_size=0.2, val_size=0.2, seed=42):
        """
        Splits the dataset into training, validation, and test sets.
        
        Args:
            db_name (str): The name of the CSV file to load data from.
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the training dataset to include in the validation split.
            seed (int): Random seed for reproducibility.
        
        Returns:
            dict: A dictionary containing the train, validation, and test datasets as TensorDatasets.
        """
        texts, labels = md.load_database(db_name)
        class_counts = pd.Series(labels).value_counts()
        try:
            texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
                texts, labels, test_size=test_size, stratify=labels, random_state=seed
            )
            
            texts_train, texts_val, labels_train, labels_val = train_test_split(
                texts_train_val, labels_train_val, test_size=val_size, stratify=labels_train_val, random_state=seed
            )
            
            train_dataset = self.encode_data(texts_train, labels_train)
            val_dataset = self.encode_data(texts_val, labels_val)
            test_dataset = self.encode_data(texts_test, labels_test)
            
            return {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
    
        except ValueError as e:
            print("Error during train_test_split:", e)
            
            problematic_classes = [label for label, count in class_counts.items() if count == 1]
            print("Classes with only one sample:", problematic_classes)
            
            sys.exit("Exiting due to class imbalance issues.")


    def create_dataloaders(self, data_dict):
        """
        Creates DataLoaders for the train, validation, and test datasets.
        
        Args:
            data_dict (dict): A dictionary containing TensorDatasets for train, validation, and test datasets.
        
        Returns:
            dict: A dictionary containing DataLoaders for train, validation, and test datasets.
        """
        train_loader = DataLoader(data_dict['train'], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(data_dict['val'], batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(data_dict['test'], batch_size=self.batch_size, shuffle=False)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
    def fit(self, train_loader, val_loader, epochs=3):
        """
        Train the BERT-based classification model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs for training.
        """
        if self.model is None:
            raise ValueError("Model is not built. Call build_model() before fitting.")
        
        # Convert DataLoader batches to TensorFlow-compatible format
        def process_batch(batch):
            input_ids, attention_masks, labels = batch
            return {
                'input_1': input_ids.numpy(), 
                'attention_mask': attention_masks.numpy()
            }, labels.numpy()
        
        # Create TensorFlow datasets
        def create_tf_dataset(loader):
            def generator():
                for batch in loader:
                    yield process_batch(batch)
            return tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    {
                        'input_1': tf.TensorSpec(shape=(None, self.max_length), dtype=tf.int64),
                        'attention_mask': tf.TensorSpec(shape=(None, self.max_length), dtype=tf.int64)
                    },
                    tf.TensorSpec(shape=(None,), dtype=tf.int64)
                )
            )
        
        train_dataset = create_tf_dataset(train_loader)
        val_dataset = create_tf_dataset(val_loader)
        
        # Configure the datasets for performance
        train_dataset = train_dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Training phase
        for epoch in range(epochs):
            for batch in train_dataset:
                inputs, labels = batch
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs, training=True)
                    loss = tf.keras.losses.binary_crossentropy(labels, outputs)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # Validation phase
            val_loss, val_accuracy = self.model.evaluate(val_dataset)
            print(f"Epoch {epoch + 1}/{epochs} completed - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

                
    def predict(self, texts):
        """
        Predict the labels for the given texts using the trained model.
        
        Args:
            texts (list): A list of texts to predict.
            
        Returns:
            list: The predicted labels.
        """
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='tf')
        predictions = self.model({
            'input_1': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        })
        return (predictions > 0.5).astype(int).numpy()

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
    model.fit(X_train, y_train, epochs = 6)
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

