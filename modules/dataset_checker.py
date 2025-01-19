from ast import Or
from cv2 import threshold
from matplotlib.pyplot import step
from numpy import average
import pandas as pd
from sympy import N
import manage_datasets as md
import os
import shutil


def get_dataframe(folder_name, file_name):
    if folder_name == "":
        file_path = os.path.join("data", file_name)
        
    else:
        file_path = os.path.join("data", folder_name, file_name)

    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, encoding='utf-8')


def get_transcription_confidences(folder_name, file_name):
    
    df = get_dataframe(folder_name, file_name)
    if df is None:
        return None
    
    confidence_rows = []
    confidence_column = df["Confidence"]
    for row in confidence_column:
        confidence_rows.append(float(row))

    return confidence_rows

def get_transcription_average_confidence(folder_name, file_name):
    confidences = get_transcription_confidences(folder_name, file_name)
    return round(sum(confidences) / len(confidences), 2)
    
def get_confidences_of_all_files(folder_name):
    
    average_confidences = []
    folder_path = os.path.join("data", folder_name)
    if not os.path.exists(folder_path):
        return None
    
    for file_name in os.listdir(folder_path):
        average_confidences.append(get_transcription_average_confidence(folder_name, file_name))

    return average_confidences

def get_average_confidence_of_all_files(folder_name):
    average_confidences = get_confidences_of_all_files(folder_name)
    return round(sum(average_confidences) / len(average_confidences), 2)

def get_average_value(values):
    return round(sum(values) / len(values)if values else 0, 2)

def get_ids_of_transcription_above_threshold(folder_name, file_name, threshold=None):
    df = get_dataframe(folder_name, file_name)
    indices_above_threshold = []
    if df is None:
        return None
    if threshold == 0:
        return list(df.index)
    elif threshold == 1:
        return []

    for i, row in df.iterrows():
        if row["Confidence"] >= threshold:
            indices_above_threshold.append(i)

    return indices_above_threshold

def get_ids_of_files_above_threshold(folder_name, threshold = None):
    '''Takes the average confidence of every file and returns the file_id of files that have more confidence that the average'''
    ids_to_include = []
    folder_path = os.path.join("data", folder_name)
    
    if not os.path.exists(folder_path):
        return None
    if threshold == 0:
        return [int(file_name[:3]) for file_name in os.listdir(folder_path)]
    elif threshold == 1:
        return []
    
    for file_name in os.listdir(folder_path):
        df = get_dataframe(folder_name, file_name)
        if df is None:
            pass
        confidences = []
        
        for index, row in df.iterrows():
            confidences.append(row['Confidence'])
        average_confidence = get_average_value(confidences)
        if average_confidence >= threshold:
            ids_to_include.append(int(file_name[:3]))
    return ids_to_include

def filter_data_by_threshold_single_file(folder_name, file_name, threshold_single_file = None):
    df = get_dataframe(folder_name, file_name)
    text = []
    ids_to_include = get_ids_of_transcription_above_threshold(folder_name, file_name, threshold_single_file)    
    for i, row in df.iterrows():
        if(i in ids_to_include):
            text.append(row['Text'])
    return text

def filter_data_by_threshold_multiple_files(folder_name, threshold_single_file = None, threshold_all_files = None, exclude_zero = False):
    folder_path = os.path.join("data", folder_name)
    if not os.path.exists(folder_path):
        return None

    reference_df = get_dataframe("", "base.csv")
    new_df = pd.DataFrame(columns=["Data", "Value"])
    ids_to_include = get_ids_of_files_above_threshold(folder_name, threshold_all_files)

    for file_name in os.listdir(folder_path):
        file_id = int(file_name[:3])
        if file_name.endswith('.csv') and file_id in ids_to_include:
            texts = filter_data_by_threshold_single_file(folder_name, file_name, threshold_single_file)
            value = reference_df.loc[reference_df['ID'] == file_id, 'Value']

            if exclude_zero:
                if value.empty or value.iloc[0] == 0:
                    continue
                else:
                    value = value.iloc[0]
            if not value.empty:
                value = value.iloc[0] 

            if texts:
                new_row = {"Data": " ".join(texts), "Value": value}

                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                print(f"Warning: No 'Text' data found in {file_name}. Skipping.")
    return new_df

def clear_directory(path):
    """
    Clears all contents of the specified directory. Creates the directory if it does not exist.
    """
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(path, exist_ok=True)


def save_dataframe(dataframe, name, path):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, f"{name}.csv")
    
    dataframe.to_csv(file_path, index=False)

    print(f"File {file_path} saved_with_success.")

def generate_csv_files(folder_name, threshold_transcription=0, threshold_files=0, steps=0.05, flag = True, control_variable = None, exclude_zero = False):
    
    base_path = os.path.join("data", "filtered_csv") if not exclude_zero else os.path.join("data", "filtered_positive_csv")

    #   Operations to perform during first iteration
    if flag:
        if control_variable is None and threshold_transcription > 0 and threshold_files == 0:
            control_variable = 'single'
            path = os.path.join(base_path, "single_file_confidence")
        elif control_variable is None and threshold_transcription == 0 and threshold_files > 0:
            control_variable = 'files'
            path = os.path.join(base_path, "multiple_file_confidence")
        else:
            control_variable = 'both'
            path = os.path.join(base_path, "all_files_confidence")
        clear_directory(path)
        flag = False

    else:

        if control_variable == 'single':
            path = os.path.join(base_path, "single_file_confidence")
            name = f"binary_dataset_single_file_{threshold_transcription:.2f}"

        elif control_variable == 'files':
            path = os.path.join(base_path, "multiple_file_confidence")
            name = f"binary_dataset_all_files_{threshold_files:.2f}"

        elif control_variable == 'both':
            path = os.path.join(base_path, "all_files_confidence")
            name = f"binary_dataset_single_and_all_files_{threshold_transcription:.2f}_{threshold_files:.2f}"

    #   Output Creation
    dataframe = filter_data_by_threshold_multiple_files(folder_name, threshold_transcription, threshold_files)

    #   Control Variable
    if control_variable == 'single':
        name = f"binary_dataset_single_file_{threshold_transcription:.2f}"
        threshold_transcription -= steps
    elif control_variable == 'files':
        name = f"binary_dataset_all_files_{threshold_files + steps:.2f}"
        threshold_files -= steps
    elif control_variable == 'both':
        name = f"binary_dataset_single_and_all_files_{threshold_transcription:.2f}_{threshold_files + steps:.2f}"
        threshold_transcription -= steps
        if threshold_transcription <= steps * 1.5:
            threshold_transcription = threshold_files
            threshold_files -= steps

    #   Saving Dataframe
    if dataframe is not None and not dataframe.empty:
        save_dataframe(dataframe, name, path)
    else:
        print(f"Warning: dataframe empty")

    #   Stop Condition
    if (threshold_transcription <= steps and threshold_files == 0) or (threshold_files <= steps and threshold_transcription == 0) or (threshold_files <= steps and threshold_transcription <= steps):
        return
    #   Recursive Call
    generate_csv_files(folder_name, threshold_transcription, threshold_files, steps, flag, control_variable, exclude_zero = exclude_zero)

    
#generate_csv_files("consolidated_csv", 0.98, 0.98, exclude_zero = False)

#generate_csv_files_with_thresholds("consolidated_csv")



