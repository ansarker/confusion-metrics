import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_score
import seaborn as sb
import pandas as pd
import os

from im2index import im2index
from confusion_matrix import confusion_metrics
from confusion_matrix import norm_confusion_metrics
from confusion_matrix import get_confusion_metrics_score
from confusion_matrix import accuracy, precision, recall, iou_conf, percentage
from confusion_metrics_to_csv import conf_mat_csv

from numba import cuda
import resource
import gc


def heatmap(conf_mat, labels, fmt, output_dir, output_name):
    data_frame = pd.DataFrame(conf_mat, index = [i for i in labels], columns = [i for i in labels])
    plt.figure(num=None, figsize=(12,6), dpi=300)
    ax = sb.heatmap(data_frame, annot=True, linewidths=.2, cmap='Blues', fmt=fmt)
    plt.title(output_name, pad=5)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted', size=12, color='g', labelpad=5)
    plt.ylabel('Actual', size=12, color='g')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.savefig(os.path.join(output_dir, output_name + '.png'))


def plot_images(fname, original, target, predict, pos):

    labels = ["Unrecognized", "Forest", "Builtup", "Water", "Farmland", "Meadow"]
    figure_dir = 'result/figure'
    conf_mat_dir = 'result/confusion-matrix'
    norm_conf_mat_dir = 'result/normalize-confusion-matrix'
    num_of_classes = 6
    
    # Convert the target and predict image into index image
    tar_ind = im2index(target)
    pre_ind = im2index(predict)

    # Confusion matrix table
    # Return a 6x6 Confusion Matrix
    conf_matrix = confusion_metrics(target=tar_ind, predict=pre_ind, num_of_classes=num_of_classes)
    normalize_conf_matrix = norm_confusion_metrics(conf_matrix=conf_matrix)
    dff_cm = pd.DataFrame(conf_matrix, index = [i for i in labels], columns = [i for i in labels])

    # Save the confusion matrix heatmap for conf_matrix
    heatmap(conf_mat=conf_matrix, labels=labels, fmt='d', output_dir=conf_mat_dir, output_name=fname)
    heatmap(conf_mat=normalize_conf_matrix, labels=labels, fmt='.4f', output_dir=norm_conf_mat_dir, output_name=fname)

    # Percentage of each classes
    percent = percentage(num_of_classes=num_of_classes, target=tar_ind)
    
    # Here to write the csv file
    conf_mat_csv(dataframe=dff_cm, percent=percent, img_title=fname.rstrip('.png'), pos=pos)
    
    # Calculate the accuracies for each index
    ac = []
    io = []
    pr = []
    rc = []
    pe = []
    
    for i in range(6):
        tp, fp, fn, tn = get_confusion_metrics_score(class_index=i, conf_metric=conf_matrix)

        acc = accuracy(tp=tp, tn=tn, fp=fp, fn=fn)
        iou_sc = iou_conf(tp=tp, fn=fn, fp=fp)
        prec = precision(tp=tp, fp=fp)
        rec = recall(tp=tp, fn=fn)
        
        ac.append(round(acc, 2))
        io.append(round(iou_sc, 2))
        pr.append(round(prec, 2))
        rc.append(round(rec, 2))
        pe.append(round(percent[i], 2))
        
        # print(f'Index: {i}\tPercentage: {pe[i]}\tAccuracy: {acc}\tIOU: {iou_sc}\tPrecision: {prec}\tRecall: {rec}')
    
    # Plot the accuracy, iou, precision and recall into figure for each classes
    f, axarr = plt.subplots(1, 3, figsize=(13, 6))
    f.suptitle(fname.rstrip('.png'), fontsize=14)

    axarr[0].imshow(original)
    axarr[0].title.set_text('Input')

    axarr[1].imshow(target)
    axarr[1].title.set_text('Ground Truth')

    axarr[2].imshow(predict)
    axarr[2].title.set_text('Output')

    unrecognized = mpatches.Patch(color="#000000", label="Unrecognized")
    forest = mpatches.Patch(color='#00FFFF', label='Forest')
    builtUp = mpatches.Patch(color='#FF0000', label='BuiltUp')
    water = mpatches.Patch(color='#0000FF', label='Water')
    farmland = mpatches.Patch(color='#00FF00', label='Farmland')
    meadow = mpatches.Patch(color='#FFFF00', label='Meadow')

    f.legend(loc='upper right', fontsize='12', handles=[unrecognized, forest, builtUp, water, farmland, meadow])

    for i in range (len(ac)):
        if io[i] == 0:
            io[i] = '-'
        if pr[i] == 0:
            pr[i] = '-'
        if rc[i] == 0:
            rc[i] = '-'

    axarr[0].plot([], [], color='#FFFFFF', label="Metric\nPercentage :\nAccuracy :\nIOU :\nPrecision: \nRecall:")
    axarr[0].plot([], [], color='#000000', label="Unrecognized\n"+str(pe[0])+"%\n"+str(ac[0])+"\n"+str(io[0])+"\n"+str(pr[0])+"\n"+str(rc[0]))
    axarr[0].plot([], [], color='#00FFFF', label="Forest\n"+str(pe[1])+"%\n"+str(ac[1])+"\n"+str(io[1])+"\n"+str(pr[1])+"\n"+str(rc[1]))
    axarr[0].plot([], [], color='#FF0000', label="BuiltUp\n"+str(pe[2])+"%\n"+str(ac[2])+"\n"+str(io[2])+"\n"+str(pr[2])+"\n"+str(rc[2]))
    axarr[0].plot([], [], color='#0000FF', label="Water\n"+str(pe[3])+"%\n"+str(ac[3])+"\n"+str(io[3])+"\n"+str(pr[3])+"\n"+str(rc[3]))
    axarr[0].plot([], [], color='#00FF00', label="Farmland\n"+str(pe[4])+"%\n"+str(ac[4])+"\n"+str(io[4])+"\n"+str(pr[4])+"\n"+str(rc[4]))
    axarr[0].plot([], [], color='#FFFF00', label="Meadow\n"+str(pe[5])+"%\n"+str(ac[5])+"\n"+str(io[5])+"\n"+str(pr[5])+"\n"+str(rc[5]))

    f.legend(loc='lower center', bbox_to_anchor=(0.485, 0.00), shadow=False, ncol=10, fontsize='12')
    plt.savefig(os.path.join(figure_dir, fname), dpi=300)

    # Free current memory
    cuda.current_context().deallocations.clear()
    gc.collect()
    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss