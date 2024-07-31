import pandas as pd
import os
import json
from nltk.corpus import wordnet as wn
from typing import Dict, List
import sys
from pathlib import Path
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
import re

import process_datasets

def get_DB_path():
    """
    Constructs and returns the path to the 'dbs' directory, which is located in the parent
    directory of the script currently being executed.

    Returns:
        str: The absolute path to the 'dbs' directory.
    """
    DB_PATH = "\\".join(os.path.dirname(os.path.abspath(__file__)).split("\\")[:-1])
    return  os.path.join(DB_PATH, "dbs")

def get_data_from_SAaDd():
    """
    Loads data from the 'Students Anxiety and Depression Dataset.xlsx'. 
    The function attempts to open and read the dataset, converting its contents
    into a list of lists.

    Returns:
        list: A list of lists containing the data from the Excel file if file is found.
        int: Returns -1 if the file is not found.

    Raises:
        FileNotFoundError: If the Excel file is not found at the specified path.
    """
    try:
        path = os.path.join(get_DB_path(), "Students Anxiety and Depression Dataset.xlsx")
        data = pd.read_excel(path)

        return data.values.tolist() # [['oh my gosh', 1.0],...]
    
    except FileNotFoundError:
        return -1
    


def get_data_from_ddrc():
    """
    Loads data from the 'Depression_dataset_reddit_cleaned.xlsx'. 
    The function attempts to open and read the dataset, converting its contents
    into a list of lists.

    Returns:
        list: A list of lists containing the data from the Excel file if file is found.
        int: Returns -1 if the file is not found.

    Raises:
        FileNotFoundError: If the Excel file is not found at the specified path.
    """
    try:
        path = os.path.join(get_DB_path(), "depression_dataset_reddit_cleaned.csv")
        dataframe = pd.read_csv(path)
        return dataframe.values.tolist() # [['oh my gosh', 1],...]
    except FileNotFoundError:
        return -1

def create_dataset(data, file_name = "composite_db.csv"):
    """
    Creates a CSV file from the provided data and saves it to the specified file within the 'dbs' directory.
    The CSV file will include an index column automatically generated by pandas, alongside two specified
    columns: 'Data' for text data and 'Value' where 1 indicates depression and 0 indicates no depression.

    Args:
        data (list of lists): Data to be saved into the CSV file. Each sublist should contain two elements:
                            the text data and its associated binary value (1 or 0).
        file_name (str, optional): Name of the file to save the data to. Defaults to "composite_db.csv".

    Returns:
        None: The function does not return any value but writes directly to a file.
    """
    path = os.path.join(get_DB_path(), file_name)

    df = pd.DataFrame(data, columns=["Data", "Value (1 is depressed / 0 is not depressed)"])
    df.to_csv(path, index=True)

def get_data_from_composite_dataset(file_name="composite_db.csv", cases=True):
    """
    Loads data from a CSV file located in the 'dbs' directory and optionally filters the rows based
    on the depression indicator (third column in the dataset). It also ensures that the second column
    entries, expected to be strings, are correctly typed.

    Args:
        file_name (str, optional): Name of the CSV file to load data from. Defaults to "composite_db.csv".
        cases (bool, optional): Filter condition to apply on the 'Value' column which represents depression.
                                If True, only rows where the 'Value' is 1 are returned.
                                If False, only rows where the 'Value' is 0 are returned.
                                Defaults to True.

    Returns:
        list: A list of rows from the CSV file that meet the specified condition and where the second
            column's entries are strings. Each row is a list itself.
        int: Returns -1 if the CSV file is not found at the specified path.

    Raises:
        FileNotFoundError: If the CSV file is not found at the specified path.
    """
    
    try:
        path = os.path.join(get_DB_path(), file_name)
        dataframe = pd.read_csv(path)

        if cases in [True, False]:  
            dataframe = dataframe[dataframe.iloc[:, 2] == int(cases)]
        return [element for index, element in enumerate(dataframe.values.tolist()) if isinstance(element[1], str)]
    except FileNotFoundError:
        return -1

def get_utils_path():
    """
    Constructs and returns the path to the 'utils.json' file, which is located in the 'dbs' directory.
    
    Returns:
        str: The absolute path to the 'utils.json' file.
    """
    return os.path.join(get_DB_path(), "utils.json")

def write_frequent_words(freq_words):
    """
    Writes the frequent words and their counts to the 'utils.json' file. If the file doesn't exist, it creates it.
    If the file is empty or the content has changed, it updates the file with the new data.
    
    Args:
        freq_words (dict): A dictionary of words and their counts.
        
    Returns:
        None
    """
    path = get_utils_path()
    try:
        if not os.path.exists(path):
            with open(path, 'w') as file:
                json.dump({"frequent_words": freq_words}, file)
        else:
            with open(path, 'r') as file:
                data = json.load(file)
            
            if "frequent_words" not in data or data["frequent_words"] != freq_words:
                data["frequent_words"] = freq_words
                with open(path, 'w') as file:
                    json.dump(data, file)
    except FileNotFoundError:
        with open(path, 'w') as file:
            json.dump({"frequent_words": freq_words}, file)

def read_frequent_words():
    """
    Reads the frequent words and their counts from the 'utils.json' file.
    
    Returns:
        dict: A dictionary of words and their counts if found, otherwise an empty dictionary.
    """
    path = get_utils_path()
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            return data.get("frequent_words", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def write_best_synsets(best_depression_synsets):
    """
    Writes the best depression synsets to the 'utils.json' file. If the file doesn't exist, it creates it.
    If the file is empty or the content has changed, it updates the file with the new data.
    
    Args:
        best_depression_synsets (dict): A dictionary of words and their best synsets.
        
    Returns:
        None
    """
    path = get_utils_path()
    try:
        if not os.path.exists(path):
            with open(path, 'w') as file:
                json.dump({"best_depression_synsets": best_depression_synsets}, file)
        else:
            with open(path, 'r') as file:
                data = json.load(file)
            
            if "best_depression_synsets" not in data or data["best_depression_synsets"] != best_depression_synsets:
                data["best_depression_synsets"] = best_depression_synsets
                with open(path, 'w') as file:
                    json.dump(data, file)
    except FileNotFoundError:
        with open(path, 'w') as file:
            json.dump({"best_depression_synsets": best_depression_synsets}, file)

def read_best_synsets():
    """
    Reads the best depression synsets from the 'utils.json' file.
    
    Returns:
        dict: A dictionary of words and their best synsets if found, otherwise an empty dictionary.
    """
    path = get_utils_path()
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            return data.get("best_depression_synsets", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
def write_tree_to_json(tree):
    """
    Writes the similarity tree to the 'utils.json' file under the 'sim_tree' key.
    If the file doesn't exist, it creates it.
    If the file is empty or the content has changed, it updates the file with the new data.
    
    Args:
        tree (dict): The tree structure to save.
        
    Returns:
        None
    """
    path = get_utils_path()
    try:
        if not os.path.exists(path):
            with open(path, 'w') as file:
                json.dump({"sim_tree": tree}, file)
        else:
            with open(path, 'r') as file:
                data = json.load(file)
            
            data["sim_tree"] = tree
            with open(path, 'w') as file:
                json.dump(data, file)
    except FileNotFoundError:
        with open(path, 'w') as file:
            json.dump({"sim_tree": tree}, file)
            
def read_tree_from_json():
    """
    Reads the similarity tree from the 'utils.json' file under the 'sim_tree' key.
    
    Returns:
        dict: The similarity tree structure if found, otherwise an empty dictionary.
    """
    path = get_utils_path()
    try:
        with open(path, 'r') as file:
            data = json.load(file)
            return data.get("sim_tree", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    
"""def convert_names_to_synsets(tree: Dict[str, Any]) -> Dict[wn.Synset, Any]:

    Converts the names of synsets in the tree to Synset objects.

    Args:
        tree (dict): The tree structure with synset names as keys.

    Returns:
        dict: The tree structure with Synset objects as keys.

    if not isinstance(tree, dict):
        return tree

    converted_tree = {}
    for synset_name, subtree in tree.items():
        try:
            synset = wn.synset(synset_name)
            converted_tree[synset] = convert_names_to_synsets(subtree)
        except wn.WordNetError:
            print(f"Error: Synset {synset_name} not found.")
    return converted_tree
    """
def print_tree(tree, level=0):
    for synset, subtree in tree.items():
        print('  ' * level + str(synset))
        if isinstance(subtree, dict):
            print_tree(subtree, level + 1)

def remove_not_understandable_sentences(lang, file_name = "composite_db.csv"):
    try:
        path = os.path.join(get_DB_path(), file_name)
        data = pd.read_csv(path)
        ids_to_remove = []
        for i, e in enumerate(data.values.tolist()):
            if not process_datasets.detect_not_correct_sentence(e[1], lang):
                ids_to_remove.append(i)
        data.drop(ids_to_remove, inplace=True)
        data.to_csv(path, index=False)
    except FileNotFoundError:
        return -1

def load_slang_dictionary(file_name = "slangs.csv"):
    """
    Loads abbreviations and their expansions from a CSV file into a dictionary.

    Args:
        file_name (str): Path to the CSV file containing abbreviations and their expansions.

    Returns:
        dict: A dictionary where the key is the abbreviation and
              the value is another dictionary with 'index' and 'expansion'.
    """
    path = os.path.join(get_DB_path(), file_name)
    df = pd.read_csv(path)
    slang_dict = {}
    for _, row in df.iterrows():
        slang_dict[row['Abbr']] = {
        'index': row.name,
        'expansion': row['Fullform']
    }
    return slang_dict

def build_slang_patterns(slang_dict):
    """
    Compiles regex patterns for all slangs to use in search and replace.

    Args:
        slang_dict (dict): Dictionary with slang abbreviations and their expansions.

    Returns:
        dict: A dictionary with slangs as keys and compiled regex patterns as values.
    """
    patterns = {}
    for slang, details in slang_dict.items():
        slang_str = str(slang)
        pattern = re.compile(r'\b{}\b'.format(re.escape(slang_str)), re.IGNORECASE)
        patterns[slang_str] = pattern
    return patterns

def replace_slangs_to_db(db_to_correct_name="composite_db.csv", slang_db="slangs.csv"):
    path = os.path.join(get_DB_path(), db_to_correct_name)
    df = pd.read_csv(path)

    slangs = load_slang_dictionary(file_name=slang_db)
    patterns = build_slang_patterns(slangs)

    def replace_slangs_in_sentence(sentence):
        """
        Replaces all slangs in a given sentence with their expansions.

        Args:
            sentence (str): The sentence to process.

        Returns:
            str: The sentence with slang replacements.
        """
        if pd.isna(sentence):
            return sentence

        if not isinstance(sentence, str):
            return sentence

        original_sentence = sentence
        for slang, pattern in patterns.items():
            if pattern.search(sentence):
                sentence = pattern.sub(slangs[slang]['expansion'], sentence)

        if original_sentence != sentence:
            print(f"Modified: Original: '{original_sentence}' -> Modified: '{sentence}'")

        return sentence

    df['Data'] = df['Data'].apply(replace_slangs_in_sentence)
    df.to_csv(path, index=False)
    print("Replaced slangs in the dataset.")



            
def remove_first_n_columns(n_col_to_remove = 4, file_name = "composite_db.csv"):
    """
    Removes the first n columns from a CSV file and saves the resulting DataFrame back to the same file.

    Args:
        n_col_to_remove (int): The number of the first n col to remove.
        file_name (str): The name of the CSV file to be processed.

    Returns:
        None
    """
    path = os.path.join(get_DB_path(), file_name)
    
    try:
        df = pd.read_csv(path)
        df = df.iloc[:, n_col_to_remove:]
        df.to_csv(path)
        print(f"Removed {n_col_to_remove} columns \t {file_name}.")
    
    except FileNotFoundError:
        print(f"Error: {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def randomize_db(file_name, new_file_name):
    """
    Copies a CSV file and randomizes the order of its rows.

    Args:
        file_name (str): The name of the original CSV file to be copied.
        new_file_name (str): The name of the new CSV file with randomized rows.

    Returns:
        None
    """
    original_path = os.path.join(get_DB_path(), file_name)
    new_path = os.path.join(get_DB_path(), new_file_name)
    
    try:
        df = pd.read_csv(original_path)
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(new_path)
        print(f"Successfully copied and randomized rows from {file_name} to {new_file_name}.")
    
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_happiness_data(file_name="text_emotion.csv"):
    path = os.path.join(get_DB_path(), file_name)
    df = pd.read_csv(path)
    
    happiness_df = df[df['sentiment'] == 'happiness']
    
    if happiness_df.empty:
        print("No data with sentiment 'happiness' found.")
        return
    
    composite_db_path = os.path.join(get_DB_path(), 'composite_db.csv')
    if os.path.exists(composite_db_path):
        composite_df = pd.read_csv(composite_db_path)
        index_column = composite_df.columns[0] 
        last_index = composite_df[index_column].max() + 1
    else:
        index_column = 'Index'
        composite_df = pd.DataFrame(columns=[index_column, 'Data', 'Value'])
        last_index = 1
    
    output_df = pd.DataFrame({
        'Data': happiness_df['content'].tolist(),
        'Value': [1] * len(happiness_df)
    })
    
    output_df[index_column] = range(last_index, last_index + len(output_df))
    
    composite_df = pd.concat([composite_df, output_df], ignore_index=True)
    
    composite_df.to_csv(composite_db_path, index=False)
    
    print(f"Data appended to {composite_db_path}")





