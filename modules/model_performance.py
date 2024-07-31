from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import manage_datasets as md
from depression_analysis_classifier import DepressionAnalysisClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def generate_weight_combinations(steps=10):
    """
    Generates a grid of weight combinations for three components summing up to 1.

    Args:
        steps (int, optional): Number of steps to divide the weight scale. Defaults to 10.

    Returns:
        list of tuples: A list of tuples, each representing a set of weights for three components.
    """
    grid = []
    for wn_weight in np.linspace(0, 1, steps):
        for fn_weight in np.linspace(0, 1 - wn_weight, steps):
            sa_weight = 1 - (wn_weight + fn_weight)
            grid.append((wn_weight, fn_weight, sa_weight))
    return grid

def optimize_percentage_to_maintain():
    """
    Optimizes the 'percentage_to_maintain' parameter for a depression analysis model using grid search.

    This function retrieves data from a composite dataset, uses it to train the model, and finds the best 
    'percentage_to_maintain' value for maximizing accuracy via grid search cross-validation.
    """    
    data = md.get_data_from_composite_dataset()
    X = [element[1] for element in data]
    y = [element[2] for element in data]
    
    model = DepressionAnalysisClassifier()
    
    param_grid = {
        'percentage_to_maintain': np.arange(0.005, 0.101, 0.005)
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print("Best accuracy: %0.2f" % grid_search.best_score_)
    print("best_parameter:", grid_search.best_params_)
    
def optimize_hyperparameters(X, y, steps):
    """
    Optimizes the hyperparameters for a depression analysis model by exploring combinations of weights and thresholds.

    Args:
        X (list): The feature set.
        y (list): The target variable (binary outcomes).
        steps (int): The number of steps to generate weight combinations, impacting the granularity of the grid search.

    Details:
        This function tests combinations of weights for three components and different threshold levels, 
        assessing their performance via cross-validation to determine the best performing parameters.
    """
    weight_combinations = generate_weight_combinations(steps = steps)  # Usa meno passaggi per ridurre il tempo di calcolo
    threshold_values = np.linspace(0, 1, 10)
    
    best_score = 0
    best_params = None
    
    for weights in weight_combinations:
        for threshold in threshold_values:
            model = DepressionAnalysisClassifier(percentage_to_maintain=0.1, threshold=threshold, wn_weight=weights[0], fn_weight=weights[1])
            score = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1))
            
            if score > best_score:
                best_score = score
                best_params = {'wn_weight': weights[0], 'fn_weight': weights[1], 'sa_weight': weights[2], 'threshold': threshold}
    
    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")
    
def get_model_accuracy():
    """
    Evaluates the accuracy of the DepressionAnalysisClassifier model by performing cross-validation.

    This function aggregates data from two conditions in the composite dataset, shuffles them for randomness,
    and calculates the cross-validated accuracy of the model, reporting the mean and the confidence interval.
    """

    model = DepressionAnalysisClassifier()    
    data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
    random.shuffle(data)
    X = [element[1] for element in data]
    y = [element[2] for element in data]
    scores = cross_val_score(model, X, y, cv = 2, n_jobs = -1)    
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
random.shuffle(data)
X = [element[1] for element in data]
y = [element[2] for element in data]
optimize_hyperparameters(X, y, 5)
optimize_percentage_to_maintain()
'''
def print_accuracy_on_n_examples(data, n):
    model = DepressionAnalysisClassifier() 
    if len(data) < n: n = len(data) 
    X = [x[1] for x in data[1:n + 1]]
    y = [x[2] for x in data[1:n + 1]]
    count = 0
    total = 0
    model.fit(X, y)
    results = model.predict(X)
    for i, e in enumerate(results):
        if(results[i] == y[i]):
            count += 1
        total += 1

    print(f"count: {count}\t total: {total}\t percentage: {100* (count / total)}")

def print_evaluation_metrics():
    """
    Trains the DepressionAnalysisClassifier model and prints evaluation metrics including 
    confusion matrix, classification report, and counts of true positives, true negatives, 
    false positives, and false negatives.
    """
    data = md.get_data_from_composite_dataset(cases=True) + md.get_data_from_composite_dataset(cases=False)
    random.shuffle(data)
    X = [element[1] for element in data]
    y = [element[2] for element in data]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = DepressionAnalysisClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
