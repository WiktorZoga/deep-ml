import numpy as np

def accuracy_score(y_true, y_pred):
    # Porównujemy tablice (dostajemy True/False) i liczymy z nich średnią
    return np.mean(y_true == y_pred)
