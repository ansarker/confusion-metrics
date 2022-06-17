import numpy as np 
from numba import jit


def confusion_metrics(target, predict, num_of_classes):
    '''
        Return a confusion metric for size of num_of_classes
    '''
    
    conf_metrics = np.zeros((num_of_classes, num_of_classes), dtype=np.int)
    for h in range(target.shape[0]):
        for w in range(target.shape[1]):
            conf_metrics[target[h,w], predict[h,w]] += 1
    return conf_metrics


def norm_confusion_metrics(conf_matrix):
    '''
        Return a normalize confusion metric.
        Divide each cell of a row by the sum of each row
    '''

    new_cf = np.zeros(conf_matrix.shape, dtype=np.float)
    for i in range(conf_matrix.shape[0]):
        new_cf[i,:] = conf_matrix[i,:] / sum(conf_matrix[i,:])

    for k in range(6):
        for l in range(6):
            if new_cf[k,l] >= 0:
                new_cf[k,l] = new_cf[k,l]
            else:
                new_cf[k,l] = 0

    return new_cf


def get_confusion_metrics_score(class_index, conf_metric):

    '''
        Return an array of tp, fp, fn, tn respectively
    '''
    conf_copy = conf_metric.copy()
    conf_copy[class_index, :] = 0
    conf_copy[:, class_index] = 0

    tp = conf_metric[class_index, class_index]
    fp = sum(conf_metric[:, class_index]) - tp
    fn = sum(conf_metric[class_index, :]) - tp
    tn = sum(sum(conf_copy))
    
    return [tp, fp, fn, tn]


def precision(tp, fp):
    if tp == 0:
        return 1
    elif tp + fp == 0:
        return 1
    else:
        return tp / (tp + fp)


def recall(tp, fn):
    if tp == 0:
        return 1
    elif tp + fn == 0:
        return 1
    else:
        return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return 1
    else:
        return (tp + tn) / (tp + tn + fp + fn)


def iou_conf(tp, fn, fp):
    if tp + fn + fp == 0:
        return 1
    else:
        return tp / (tp + fn + fp)


def percentage(num_of_classes, target):
    total = target.shape[0] * target.shape[1]
    percentage = np.arange(0, num_of_classes, dtype=np.int)
    
    for i in target.ravel():
        percentage[i] += 1

    return (percentage/total) * 100