import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt


def heatmap(predict : pd.DataFrame, label : pd.DataFrame) -> None:
    df = pd.DataFrame({'Predict': predict, 'Label': label})
    data = pd.crosstab(df['Predict'], df['Label'])

    hm = sn.heatmap(data = data, annot = True, cmap='Blues', fmt='g')
    #plt.show()
    return hm

def hist_data_column(X: pd.DataFrame) -> None:
    for col in X.columns:
        plt.figure(figsize=(2,2), dpi=80)
        sn.histplot(data=X, x=col, bins='auto')
        plt.show()

def barChart(Y : np.ndarray) -> None:
    unique, counts = np.unique(Y, return_counts=True)
    result = dict(zip(unique, counts))
    plt.bar(result.keys(), result.values())
    plt.show()
