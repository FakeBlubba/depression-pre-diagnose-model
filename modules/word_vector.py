import pandas as pd
#from transformers import BertTokenizer, BertModel
#from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
import sys
import string
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
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


df = preprocess_data("composite_db.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df['Processed_Data'], df['Label'], test_size=0.2, random_state=42)

train_dataset = encode_data(train_texts, train_labels, tokenizer)
val_dataset = encode_data(val_texts, val_labels, tokenizer)