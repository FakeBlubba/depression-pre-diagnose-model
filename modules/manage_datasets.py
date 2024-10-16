import pandas as pd
import os
import json
from typing import Dict, List
import sys
from pathlib import Path
import re
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


def load_database(db_name = "composite_db.csv"):
    df = get_data_from_composite_dataset(db_name)
    
    
    texts = [x[0] for x in df[1:]]
    labels = [x[1] for x in df[1:]]
    return texts, labels


def create_dataset(data, file_name="composite_db.csv"):
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
    for dt in data:
        if all(len(row) == 2 for row in dt):
            df = pd.DataFrame(dt, columns=["Data", "Value"])
            df.to_csv(path, index=False)
            print(f"Data saved to {path}")
        else:
            print("Error: Data format is incorrect. Each element must be a list with exactly two elements.")

def get_data_from_composite_dataset(file_name="composite_db.csv", cases=None):
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

        if cases is not None:
            dataframe = dataframe[dataframe['Value'] == int(cases)]
        
        return [element for element in dataframe.values.tolist() if isinstance(element[0], str)]
    
    except FileNotFoundError:
        return -1

def get_utils_path():
    """
    Constructs and returns the path to the 'utils.json' file, which is located in the 'dbs' directory.
    
    Returns:
        str: The absolute path to the 'utils.json' file.
    """
    return os.path.join(get_DB_path(), "utils.json")


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
    """
    Loads a dataset containing text and sentiment data, filters the rows where the sentiment is 'happiness',
    and appends that data to a composite database. If the composite database already exists, the function adds 
    the new data starting from the last index; otherwise, it creates a new composite dataset.

    Args:
        file_name (str): The name of the file containing the sentiment data. Default is 'text_emotion.csv'.
    
    Returns:
        None: The function saves the updated composite dataset to the 'composite_db.csv' file.
    """
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

def get_data_from_mdet(file_name="MultiLabeled_Depression_English_57000_Tweet.csv"):
    """
    Loads data from the specified CSV file and prepares it for insertion into the database.
    The function excludes rows with a label of 0, transforms labels of -1 into 0, 
    and keeps labels of 1 unchanged. The prepared data is returned as a list of lists.

    Args:
        file_name (str): The name of the CSV file to load. Default is "MultiLabeled_Depression_English_57000_Tweet.csv".
    
    Returns:
        list of lists: List of lists with filtered and transformed data, where each sublist contains
                        the text data and its associated binary value (1 or 0).
    """
    path = os.path.join(get_DB_path(), file_name)
    df = pd.read_csv(path)

    df = df[df['label'] != 0]

    df['label'] = df['label'].replace(-1, 0)

    data_list = df[['text', 'label']].values.tolist()

    return data_list

def append_data_to_composite(prepared_df):
    """
    Appends a DataFrame containing new data to an existing composite dataset stored in 'composite_db.csv'.
    If the composite dataset doesn't exist, a new one is created with an 'Index' column, and the data is added.
    
    Args:
        prepared_df (pd.DataFrame): The DataFrame containing new data to be appended. It must have 'Data' 
        and 'Value' columns, representing the textual content and the associated label, respectively.
        Returns:
        None: The function updates and saves the composite dataset.
    """
    composite_db_path = os.path.join(get_DB_path(), 'composite_db.csv')
    if os.path.exists(composite_db_path):
        composite_df = pd.read_csv(composite_db_path)
        index_column = composite_df.columns[0]
        last_index = composite_df[index_column].max() + 1
    else:
        index_column = 'Index'
        composite_df = pd.DataFrame(columns=[index_column, 'Data', 'Value'])
        last_index = 1

    prepared_df[index_column] = range(last_index, last_index + len(prepared_df))
    
    composite_df = pd.concat([composite_df, prepared_df], ignore_index=True)
    
    composite_df.to_csv(composite_db_path, index=False)
    
    print(f"Data appended to {composite_db_path}")

    



