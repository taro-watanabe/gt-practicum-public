import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils import plot_series, train_test_data

warnings.filterwarnings("ignore")


def train_model(data, k):
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    model = KMeans(n_clusters=k, random_state=6748)
    model.fit(X)

    d = data.copy()
    d["cluster"] = model.labels_

    print(model.cluster_centers_)

    return model, d, model.cluster_centers_


def _predict(model, new_data):
    n = new_data.copy()
    n["cluster"] = model.predict(new_data)

    return n


def _aggregate(df, granularity, col_method_dict):
    d = df.copy()
    d.index = pd.to_datetime(d.index, infer_datetime_format=True)
    d = d.resample(granularity).agg(col_method_dict)

    return d


def _calculate_distance(centers, n_predicted):
    n = n_predicted.copy()
    m = n_predicted.copy()
    m.drop(columns=["cluster"], inplace=True)
    # we need to standardize the data before calculating the distance
    scaler = StandardScaler()
    m = scaler.fit_transform(m)
    # calculate distance of each point to each cluster
    for i in range(len(centers)):
        # create a boolean mask for the datapoints in cluster i
        mask = n["cluster"] == i
        # calculate the Euclidean distance between each
        # datapoint in cluster i and its centroid
        distances = np.linalg.norm(m[mask] - centers[i], axis=1)
        # pairwise distance bertween first two columns
        # of m and centers[i][0:2] and iterate

        # assign the distances to a new column "distance" in the dataframe
        n.loc[mask, "distance"] = distances
    return n


def fit_predict_plot(train, test, k, save_plot=True, fig_prefix=""):
    print(f"Training model with k={k}...")
    img_dir = "plots/kmeans/"
    base_image_name = fig_prefix + f"K-means_{k}"
    model, train, centers = train_model(train, k)
    test_predicted = _predict(model, test)
    test_predicted = _calculate_distance(centers, test_predicted)

    trunced = _aggregate(test_predicted, "D", {"distance": ["mean", "median", "max"]})
    trunced.columns = trunced.columns.map("_".join)

    # subtract minimum 5%
    trunced["distance_adjusted"] = np.array(
        trunced["distance_mean"],
    ) - np.percentile(
        trunced["distance_mean"],
        5,
    )
    trunced["distance_max_smoothed"] = (
        trunced["distance_max"].rolling(window=12, center=False).mean()
    )
    plot_series(
        trunced,
        [
            "distance_mean",
            "distance_median",
            "distance_max",
            "distance_max_smoothed",
        ],
        figname=f"{img_dir}daily_{base_image_name}.png",
        save=save_plot,
    )

    trunced_3_hourly = _aggregate(
        test_predicted, "3H", {"distance": ["mean", "median", "max"]}
    )
    trunced_3_hourly.columns = trunced_3_hourly.columns.map("_".join)
    trunced_3_hourly["distance_smooth_max"] = (
        trunced_3_hourly["distance_max"].rolling(window=100, center=False).mean()
    )

    plot_series(
        trunced_3_hourly,
        [
            "distance_mean",
            "distance_median",
            "distance_max",
            "distance_smooth_max",
        ],
        figname=f"{img_dir}3_hourly_{base_image_name}.png",
        save=save_plot,
    )

    return trunced, trunced_3_hourly


if __name__ == "__main__":
    train, test = train_test_data()
    # filter out train where timestamp is after min of test timestamp
    train_before = train[train.index < test.index.min()]
    train_after = train[train.index >= test.index.min()]
    test_before = test[test.index < train_after.index.min()]
    test_after = test[test.index >= train_after.index.min()]

    for k in range(2, 11):
        t_b, t_b_3 = fit_predict_plot(train_before, test_before, k)
        t_a, t_a_3 = fit_predict_plot(train, test_after, k)

        t = pd.concat([t_b, t_a])
        t_3 = pd.concat([t_b_3, t_a_3])
        t["distance_smooth_max"] = (
            t["distance_max"].rolling(window=12, center=False).mean()
        )
        t_3["distance_smooth_max"] = (
            t_3["distance_max"].rolling(window=60, center=False).mean()
        )

        plot_series(
            t,
            [
                "distance_mean",
                "distance_median",
                "distance_max",
                "distance_smooth_max",
            ],
            figname=f"plots/kmeans/ADJ_daily_K-means_{k}.png",
            title=f"Daily K-means: k={k}",
        )

        plot_series(
            t_3,
            [
                "distance_mean",
                "distance_median",
                "distance_max",
                "distance_smooth_max",
            ],
            figname=f"plots/kmeans/ADJ_3_hourly_K-means_{k}.png",
            title=f"3-hourly K-means: k={k}",
        )
