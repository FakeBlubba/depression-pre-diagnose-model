from pathlib import Path
import sys
modules_path = Path(__file__).parent / 'modules'
sys.path.append(str(modules_path))
from classifier import BertDepressionClassifier
import manage_datasets as md
import dataset_checker

def main():
    try:
        classifier = BertDepressionClassifier()
        d = classifier.get_data_processed("composite_db.csv")
        X_train, X_test, y_train, y_test = classifier.train_data(d)
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