
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # For CUDA
# pip3 install torch torchvision torchaudio # For no GPU

import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import sys
import string
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
#from official.nlp import optimization
from transformers import BertTokenizer

from sklearn.model_selection import train_test_split

modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))

import model_performance
import manage_datasets as md
import process_datasets
import score_generator
import audio_manager
import dataset_checker
from depression_analysis_classifier import DepressionAnalysisClassifier


#tf.get_logger().setLevel('ERROR')
def preprocess_text(text, use_wordnet=False):
    """
    Preprocesses the given text by tokenizing, removing stopwords, and applying stemming.

    Args:
        text (str): The text to preprocess.
        use_wordnet (bool): Whether to use WordNet for additional processing.

    Returns:
        list: A list of processed tokens.
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

            
    return "".join(tokens)

def preprocess_data(file_name):
    '''
    Processes the dataset by reading from a CSV file and applying preprocessing to each row.

    Args:
        file_name (str): The name of the CSV file to process.

    Returns:
        DataFrame: A pandas DataFrame with an additional column 'Processed_Data' containing preprocessed text.
    '''
    df = pd.read_csv(file_name)
    df['Processed_Data'] = df['Data'].apply(preprocess_text)
    return df

def preprocess_data(file_name, use_wordnet=False):
    """
    Processes the dataset by reading from a CSV file and applying preprocessing to each row.

    Args:
        file_name (str): The name of the CSV file to process.
        use_wordnet (bool): Whether to use WordNet for additional processing.

    Returns:
        list: A list of lists where each inner list contains processed tokens for a row.
    """
    path = os.path.join(md.get_DB_path(), file_name)
    df = pd.read_csv(path)
    
    sentences = df['Data'].tolist()
    
    processed_sentences = [preprocess_text(sentence, use_wordnet) for sentence in sentences]
    
    return processed_sentences

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(texts, labels, tokenizer, max_length=128):
    '''
    Encodes the texts and labels for BERT input.

    Args:
        texts (list): A list of text samples to encode.
        labels (list): A list of labels corresponding to the text samples.
        tokenizer (BertTokenizer): The BERT tokenizer to use for encoding.
        max_length (int): The maximum length of the tokenized sequences.

    Returns:
        TensorDataset: A dataset containing input IDs, attention masks, and labels.
    '''
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_masks = torch.tensor(encodings['attention_mask'])
    labels = torch.tensor(labels.values)
    return TensorDataset(input_ids, attention_masks, labels)

'''preprocessed_sentence = preprocess_text("This is an example sentence.", True)
preprocessed_sentence2 = preprocess_text("This is an example sentence.", False)

print("Processata con WordNet: ", preprocessed_sentence, "\nProcessata senza WordNet: ", preprocessed_sentence2)'''

def split_sets(db_name="composite_db.csv", test_size=0.2, val_size=0.2, seed = 42):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        db_name (str): The name of the CSV file to load data from.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training dataset to include in the validation split.

    Returns:
        dict: A dictionary containing the train, validation, and test datasets as TensorDatasets.
    """
    data = md.get_data_from_composite_dataset(file_name=db_name)
    
    if data == -1:
        raise FileNotFoundError(f"{db_name}: Not Found!")
    
    df = pd.DataFrame(data, columns=['Data', 'Value'])
    
    texts = df['Data']
    labels = df['Value']
    
    texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train_val, labels_train_val, test_size=val_size, stratify=labels_train_val, random_state=seed
    )
    
    train_dataset = encode_data(texts_train, labels_train, tokenizer)
    val_dataset = encode_data(texts_val, labels_val, tokenizer)
    test_dataset = encode_data(texts_test, labels_test, tokenizer)
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
def create_dataloaders(data_dict, batch_size=32):
    """
    Creates DataLoaders for the train, validation, and test datasets.

    Args:
        train_dataset (TensorDataset): The training dataset.
        val_dataset (TensorDataset): The validation dataset.
        test_dataset (TensorDataset): The test dataset.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        dict: A dictionary containing DataLoaders for train, validation, and test datasets.
    """
    train_loader = DataLoader(data_dict['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_dict['val'], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_dict['test'], batch_size=batch_size, shuffle=False)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

d = split_sets()
loaded = create_dataloaders(d, batch_size=32)
for batch in loaded["train"]:
    input_ids, attention_masks, labels = batch
    # Ogni frase ha diversi ID, non è detto che corrispondano a ogni parola
    print(f"Input IDs: {input_ids}")    
    # Ogni frase ha diversi attention mask (1), servono a dire se il singolo token deve essere considerato oppure se no (0)
    print(f"Attention Masks: {attention_masks}")  
    # Le etichette del database
    print(f"Labels: {labels}")
    break  

#train_texts, val_texts, train_labels, val_labels = train_test_split(df['Data'], df['Value'], test_size=0.2, random_state=42)
