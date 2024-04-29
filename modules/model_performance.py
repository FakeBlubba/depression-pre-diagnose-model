from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import manage_datasets as md
from depression_analysis_classifier import DepressionAnalysisClassifier

def generate_weight_combinations(steps=10):
    grid = []
    for wn_weight in np.linspace(0, 1, steps):
        for fn_weight in np.linspace(0, 1 - wn_weight, steps):
            sa_weight = 1 - (wn_weight + fn_weight)
            grid.append((wn_weight, fn_weight, sa_weight))
    return grid

def optimize_percentage_to_maintain():
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
    model = DepressionAnalysisClassifier()    
    data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
    random.shuffle(data)
    X = [element[1] for element in data]
    y = [element[2] for element in data]
    scores = cross_val_score(model, X, y, cv = 5, n_jobs = -1)    
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

data = md.get_data_from_composite_dataset(cases = True) + md.get_data_from_composite_dataset(cases = False)
random.shuffle(data)
X = [element[1] for element in data]
y = [element[2] for element in data]
optimize_hyperparameters(X, y, 5)
optimize_percentage_to_maintain()