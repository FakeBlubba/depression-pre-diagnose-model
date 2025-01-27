from pathlib import Path
import sys
from modules.manage_datasets import get_test_set_ids
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
from classifier import BertDepressionClassifier
import manage_datasets as md
import dataset_checker as dc
import numpy as np


import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


from sklearn.metrics import confusion_matrix, classification_report
def main():
    try:
        '''# Download latest version
        path = kagglehub.model_download("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/cmlm-en-large")

        print("Path to model files:", path)'''
        classifier = BertDepressionClassifier()
        X_train, X_test, y_train, y_test = md.get_train_and_test_set("binary_dataset.csv")

        #d = classifier.get_formatted_data("binary_dataset.csv")    
        
        
        cv_type = input("Select cross-validation type (loocv/simple/no): ").strip().lower()
        
        if cv_type == "loocv":
            results = classifier.leave_one_out_cv(X_train + X_test, y_train + y_test, 2)
            print("\nLOOCV Results:")
            for metric, value in results.items():
                print(f"{metric.capitalize()}: {value:.4f}")  
        
        elif cv_type == "simple":
            X = np.array(d["Data"])
            y = np.array(d["Value"])
            n_splits = int(input("Enter the number of splits for simple CV (e.g., 5): "))
            results = classifier.simple_cross_validation(X, y, n_splits=n_splits) 
            print("\nSimple Cross-Validation Results:")
            for metric, value in results.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            return  


        elif cv_type == "no":
            model = classifier.create_model()
            model = classifier.compile_model(model)
            model.fit(X_train, y_train, epochs=3)
            model.evaluate(X_test, y_test)
        
            y_predicted = model.predict(X_test)
            y_predicted = y_predicted.flatten()
            y_predicted = np.where(y_predicted > 0.5, 1, 0)
            print(y_predicted)
        
            # Metrics
            cm = confusion_matrix(y_test, y_predicted)
            print(f"Confusion Matrix\n{cm}\n{classification_report(y_test, y_predicted)}")
        
        else:
            print("Invalid option. Please select 'loocv', 'simple', or 'no'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

main()
