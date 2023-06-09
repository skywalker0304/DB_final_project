import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import missingno as msno
import math
from preprocess import encode


def heatmap(predict: pd.DataFrame, label: pd.DataFrame):
    df = pd.DataFrame({'Predict': predict, 'Label': label})
    data = pd.crosstab(df['Predict'], df['Label'])

    hm = sn.heatmap(data=data, annot=True, cmap='Blues', fmt='g')
    #plt.show()
    return plt.gcf()


def barChart(Y: np.ndarray):
    unique, counts = np.unique(Y, return_counts=True)
    result = dict(zip(unique, counts))
    plt.bar(result.keys(), result.values())
    # plt.show()
    return plt.gcf()


def missingMap(df: pd.DataFrame):
    msno.matrix(df)
    return plt.gcf()


def numeric_feature_barchart(df: pd.DataFrame, nf: list):

    encode(df)
    n = 5
    plt.figure(figsize=[15, 4*math.ceil(len(nf)/n)])
    for i in range(len(nf)):
        plt.subplot(math.ceil(len(nf)/3), n, i+1)
        sn.distplot(df[nf[i]], hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255,255,255])/255))
    plt.tight_layout()
    return plt.gcf()
