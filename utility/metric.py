from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def evaluation(y_true, y_pred,is_sigmoid):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    a = metrics.accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return a, p, r, f


