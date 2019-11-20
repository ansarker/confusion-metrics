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


def get_confusion_metrics_score(class_index, conf_metric):

    '''
        Return an array of tp, fp, fn, tn respectively
    '''
    
    hor_sum = sum(conf_metric[class_index, :])
    ver_sum = sum(conf_metric[:, class_index])
    total = sum(sum(conf_metric))

    tp = conf_metric[class_index, class_index]
    fp = sum(conf_metric[:, class_index]) - tp
    fn = sum(conf_metric[class_index, :]) - tp
    tn = total - (hor_sum + ver_sum)
    
    return [tp, fp, fn, tn]


def precision(tp, fp):
    if tp == 0:
        return 0
    elif tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)


def recall(tp, fn):
    if tp == 0:
        return 0
    elif tp + fn == 0:
        return 0
    else:
        return tp / (tp + fn)


def accuracy(tp, tn, fp, fn):
    if tp + tn + fp + fn == 0:
        return 0
    else:
        return (tp + tn) / (tp + tn + fp + fn)


def iou_conf(tp, fn, fp):
    if tp + fn + fp == 0:
        return 0
    else:
        return tp / (tp + fn + fp)


def percentage(num_of_classes, target):
    total = target.shape[0] * target.shape[1]
    percentage = np.arange(0, num_of_classes, dtype=np.int)
    
    for i in target.ravel():
        percentage[i] += 1

    return (percentage/total) * 100