import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import warnings
from utils import plot_series, train_test_data


def train_single_model(data):
    models = {}
    # if any cell is empty in data
    for feature in data.columns:
        model = ARIMA(data[feature], order=(2, 1, 0))
        model_fit = model.fit()
        models[feature] = model_fit
    return models


def calculate_anomaly_score(models, row):
    residuals = []
    for feature in row.index:
        forecasted_value = models[feature].forecast(steps=1).iloc[0]
        actual_value = row[feature]
        residual = actual_value - forecasted_value
        residuals.append(residual)
    anomaly_score = np.mean(np.abs(residuals))
    return anomaly_score

if __name__ == "__main__":
    train, test = train_test_data()
    res = test.copy()
    anomaly_scores = []
    models = train_single_model(train)
    print(res.head())
    # for i in range(len(test)):
    for i in range(20):
        # print progress
        print(f"Processing row {i} of {len(test)}, {round(i/len(test)*100, 2)}%")
        row = test.iloc[i]
        models = train_single_model(train)
        anomaly_score = calculate_anomaly_score(models, row)
        res.loc[i, "anomaly_score"] = anomaly_score
        print(anomaly_score)
        anomaly_scores.append(anomaly_score)
        train = train._append(row)
        try:
            for feature in row.index:
                models[feature] = models[feature].append([row[feature]], refit=True)
        except:
            continue

    plot_series(res, "anomaly_score")