import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from confusion_matrix import norm_confusion_metrics
from confusion_matrix import get_confusion_metrics_score
from confusion_matrix import accuracy, iou_conf, precision, recall
from plotting import heatmap

def total_conf_mat(conf_mat_dataframe):
    labels = ["Unrecognized", "Forest", "Builtup", "Water", "Farmland", "Meadow"]
    conf_mat = []

    for col in range(8, conf_mat_dataframe.shape[1]):
        col_name = conf_mat_dataframe.columns[col]
        conf_mat_val = sum(conf_mat_dataframe[col_name])
        conf_mat.append(conf_mat_val)

    conf_mat = np.reshape(conf_mat, (-1, 6))

    df_cm = pd.DataFrame(conf_mat, index = [i for i in labels], columns = [i for i in labels])
    df_cm.to_csv('result/csvs/total-conf-mat.csv', header=True)
    norm_conf = norm_confusion_metrics(conf_mat)

    heatmap(conf_mat=conf_mat, labels=labels, fmt='d', output_dir='result/csvs', output_name='total-conf-mat')
    heatmap(conf_mat=norm_conf, labels=labels, fmt='.4f', output_dir='result/csvs', output_name='norm-total-conf-mat')

    output_metrics = np.zeros((5,7), dtype=np.object)
    output_metrics[0,:] = np.concatenate((['Metrics'], labels))
    output_metrics[1:,0] = ['Accuracy', 'IoU', 'Precision', 'Recall']

    mat = np.zeros((4,6))

    for i in range(6):
        tp, fp, fn, tn = get_confusion_metrics_score(class_index=i, conf_metric=conf_mat)
        
        acc = accuracy(tp=tp, tn=tn, fp=fp, fn=fn)
        iou_sc = iou_conf(tp=tp, fp=fp, fn=fn)
        prec = precision(tp=tp, fp=fp)
        rec = recall(tp=tp, fn=fn)
        
        mat[0,i] = acc
        mat[1,i] = iou_sc
        mat[2,i] = prec
        mat[3,i] = rec

        print(f'{labels[i]}')
        print(f'\tAccuracy: {acc}')
        print(f'\tIoU: {iou_sc}')
        print(f'\tPrecision: {prec}')
        print(f'\tRecall: {rec}')

    output_metrics[1:,1:] = mat
    output_mat = pd.DataFrame(output_metrics)
    output_mat.to_csv('result/csvs/overall-performance.csv', header=False, index=False)