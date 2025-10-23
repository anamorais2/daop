import numpy as np

def get_metrics(y_true, y_pred):
    ypred = np.argmax(y_pred,axis=1) 
    ytrue = np.argmax(y_true,axis=1)

    return np.sum(ytrue == ypred)/len(ytrue)
