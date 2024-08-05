import sys
import random
import nltk
from pathlib import Path
from nltk.corpus import stopwords
import pandas as pd

modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))

import model_performance
import manage_datasets as md
import process_datasets
import score_generator
import audio_manager
import dataset_checker
from depression_analysis_classifier import DepressionAnalysisClassifier


def main():   
    
    # Remove comments to start from beginning
    #datasets_list = [md.get_data_from_SAaDd(), md.get_data_from_ddrc()]

    #formatted_datasets = process_datasets.format_datasets(datasets_list)

    #md.create_dataset(formatted_datasets)    
    
    # audio recording and transcription
    #audio_manager.save_audio_file("record.wav")
    #input = audio_manager.transcription = audio_manager.transcript_audio("record.wav")
    #md.remove_not_understandable_sentences("en", "composite_db_randomized.csv")
    #md.replace_slangs_to_db("composite_db.csv")
    #md.replace_slangs_to_db("composite_db_randomized.csv")
    #md.remove_first_n_columns(4, "composite_db.csv")
    #md.load_mlde()
    #md.load_happiness_data()
    #md.randomize_db("composite_db.csv", "composite_db_randomized.csv")
    #md.remove_first_n_columns(2, "composite_db_randomized.csv")
    #data = pd.read_csv("dbs\\composite_db_randomized.csv")
    #data = data.values.tolist()
    #model_performance.print_accuracy_on_n_examples(data, 100)
    #md.process_dataset(db_name, "slangs.csv") # Toglie slang
    
    # Creazione DB
    '''
    dataset_list = [md.get_data_from_ddrc(), md.get_data_from_SAaDd(), md.get_data_from_mdet()]
    md.create_dataset(dataset_list)

    md.remove_not_understandable_sentences("en", db_name)'''
    db_name = "composite_db.csv"
    dataset_checker.check_balanced_db(db_name, 0.16)



    '''data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
    random.shuffle(data)
    X = [element[1] for element in data[:-1]]
    y = [element[2] for element in data[:-1]]
    model_performance.print_evaluation_metrics()'''
    #model.fit(X, y)
    #response = model.predict(input)
    #"""print("", response, "\t-Target: \t", data[-1][2]) """
    '''try:

        #model_performance.get_model_accuracy()
        return
    except Exception as e:
        print("error: ", e)'''
if __name__ == "__main__":
    main()