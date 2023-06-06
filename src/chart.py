import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


def heatmap(predict : pd.DataFrame, label : pd.DataFrame) -> None:
    df = pd.DataFrame({'Predict': predict, 'Label': label})
    data = pd.crosstab(df['Predict'], df['Label'])

    hm = sn.heatmap(data = data, annot = True, cmap='Blues', fmt='g')
    #plt.show()
    return plt.gcf()


def barChart(Y : np.ndarray) -> None:
    unique, counts = np.unique(Y, return_counts=True)
    result = dict(zip(unique, counts))
    plt.bar(result.keys(), result.values())
    # plt.show()
    return plt.gcf()
