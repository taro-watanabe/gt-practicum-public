import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


def plot_series(df, col, figtype="line", figname="img.png", save=True, title=None):
    fig, ax = plt.subplots(figsize=(20, 12))
    df[col].plot(ax=ax, kind=figtype)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(col)
    plt.show()

    # remove x-tick
    ax.set_xticklabels([])

    if save:
        fig.savefig(figname)
    plt.clf()

    return None


def load_data():
    # data ingestion
    df_1 = pd.read_csv("d1.csv")
    df_2 = pd.read_csv("d2.csv")

    # sample.csv is availble just for columnal reference.

    return df_1, df_2


def _get_learning_time_df_1():
    df_1, _ = load_data()
    a = df_1[df_1["iot_id"].str.contains("fft_100")]["timestamp"].unique()
    # conatins 'learning_percentage' in iot_id
    b = df_1[df_1["iot_id"].str.contains("learning_percentage")]["timestamp"].unique()

    return np.intersect1d(a, b)


def train_test_data():
    """
    Returns two dataframes ready for training and testing
    """
    df_1, df_2 = load_data()

    # content hidden for confidentiality

    return df_1, df_2
