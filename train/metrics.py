import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.metrics import f1_score
import numpy as np


def get_metric(metric):
    if metric == 'auc':
        return roc_auc_score
    elif metric == 'rmse':
        return rmse
    else:
        raise ValueError('Metric Error.')


def rmse(label,pred):
    result = mean_squared_error(label,pred)
    return math.sqrt(result)


def compute_score(pred, label, metric_f, task_num):
    pred_val = []
    label_val = []
    for i in range(task_num):
        pred_val_i = []
        label_val_i = []
        for j in range(len(pred)):
            if label[j][i] != -1:
                pred_val_i.append(pred[j][i])
                label_val_i.append(label[j][i])
        pred_val.append(pred_val_i)
        label_val.append(label_val_i)

    result = []
    for i in range(task_num):
        if all(one == 0 for one in label_val[i]) or all(one == 1 for one in label_val[i]):
            result.append(float('nan'))
            continue
        if all(one == 0 for one in pred_val[i]) or all(one == 1 for one in pred_val[i]):
            result.append(float('nan'))
            continue
        re = metric_f(label_val[i], pred_val[i])
        result.append(re)

    result = np.nanmean(result)

    return result


def accuracy_TU(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc




def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 
    
    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')

  
def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc
