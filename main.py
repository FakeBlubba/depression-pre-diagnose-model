import sys
import random
import nltk
from pathlib import Path
from nltk.corpus import stopwords

modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))

import model_performance
import manage_datasets as md
import process_datasets
import score_generator
import audio_manager
from depression_analysis_classifier import DepressionAnalysisClassifier


def main():    
    
    # Remove comments to start from beginning
    #datasets_list = [md.get_data_from_SAaDd(), md.get_data_from_ddrc()]

    #formatted_datasets = process_datasets.format_datasets(datasets_list)

    #md.create_dataset(formatted_datasets)    
    
    # audio recording and transcription
    #audio_manager.save_audio_file("record.wav")
    #input = audio_manager.transcription = audio_manager.transcript_audio("record.wav")
    """model = DepressionAnalysisClassifier()    
    data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
     random.shuffle(data)
    input = [data[-1][1]]
    print(input, data[-1][2])
    X = [element[1] for element in data[:-1]]
    y = [element[2] for element in data[:-1]]
    model.fit(X, y)
    response = model.predict(input)
    print("", response, "\t-Target: \t", data[-1][2]) """
    # TODO PENSARE DI AGGIUNGERE DATASET SULLA FELICITA'
    try:
        model_performance.get_model_accuracy()
    except Exception as e:
        print("error: ", e)
if __name__ == "__main__":
    main()