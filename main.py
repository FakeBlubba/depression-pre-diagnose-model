from pathlib import Path
import sys

from modules.manage_datasets import get_test_set_ids
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
from classifier import BertDepressionClassifier
import manage_datasets as md
import dataset_checker as dc
import numpy as np


from sklearn.metrics import confusion_matrix, classification_report
def main():
    try:
        '''        # Download latest version
        path = kagglehub.model_download("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-en-large")

        print("Path to model files:", path)'''
        classifier = BertDepressionClassifier()
        d = classifier.get_data_processed("filtered_positive_csv\\all_files_confidence\\binary_dataset_single_and_all_files_0.08_0.13.csv")
        X_train, X_test, y_train, y_test = classifier.train_data(d)
        tests = md.get_test_set_texts()
        X_test = tests[0]
        y_test = tests[1]
        
        model = classifier.create_model()
        model = classifier.compile_model(model)
        model.fit(X_train, y_train, epochs = 3)
        model.evaluate(X_test, y_test)
    
        y_predicted = model.predict(X_test)
        y_predicted = y_predicted.flatten()
        y_predicted = np.where(y_predicted > 0.5, 1, 0)
        print(y_predicted)
    
        # Metrics
        cm = confusion_matrix(y_test, y_predicted)
        print(f"Confusion Matrix\n{cm}\n{classification_report(y_test, y_predicted)}")
        

    except Exception as e:
        print(f"An error occurred: {e}")

main()