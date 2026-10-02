import numpy as np

def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    allp = np.sum(y_pred == 1)
    
    if allp == 0:
        return 0.0
        
    return tp / allp
