# --*-- coding: utf-8 --*--
"""

"""

import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def cf_matrix(y_true, y_predict, labels, error=True):
    """
    Recalculate the confusion matrix for labeling error,
    right predicted error is set to 0,
    the others are calculated based on percentages of the given numbers of gold labels
    except for the matched predicted labels
    :param:
        :y_true: (list) true labels
        :y_predict: (list) predicted labels
        :labels: list of given labels/classes

    :return: (np.array) new confusion matrix (transposed), with gold labels on x axis, predicted labels on y axis
                (normal confusion matrices are reversed)
    """
    matrix = confusion_matrix(y_true, y_predict, labels=labels)
    label_sum = matrix.sum(axis=1)

    if error:
        for i in range(len(label_sum)):
            label_sum[i] -= matrix[i, i]
            matrix[i, i] = 0

    for i, (vec, lsum) in enumerate(zip(matrix, label_sum)):
        if lsum != 0:
            matrix[i] = np.around(vec * 100 / lsum, decimals=2)

    return matrix.T


def cf_matrix_plot(matrix, labels, size, save=False, outdir='./', err=True):
    """
    Confusion matrix plot for labeling error
    :param:
        :matrix: (np.array) datas for confusion matrix (in form of 2-dimensional list mxm)
        :labels: (list) list of labels
        :size: (tuple) resize this figure
        :emb: (str) type of embedding
        :save: (bool) whether to save this figure
        :outdir: (str) where to save output
    """

    # set x axis on top
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # set figure size
    plt.figure(figsize=size)

    # plot confusion matrix
    df_cm = pd.DataFrame(matrix, labels, labels)
    sns = sn.heatmap(df_cm, cbar=False, annot=True, annot_kws={'size': 16}, alpha=0.9, fmt='d')

    # add label to x and y axis
    tick_marks = np.arange(len(labels)) + 0.5
    plt.xticks(tick_marks, labels, va='center')
    plt.yticks(tick_marks, labels, va='center')

    plt.tight_layout()

    name = None
    if save:
        error = '_error' if err else ''
        name = f'{outdir}/label_confusion_matrix_{error}.png'
        plt.savefig(name, dpi='figure')
    else:
        plt.show()
    plt.close()

    if name:
        print(f'Check {name}\n')