import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def load_data(data_path):
    # Load the data
    data = np.genfromtxt(data_path, delimiter=',')[:,:-1]
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.transpose((0,2,3,1))
    return data

if __name__ == '__main__':
    data = load_data("data/Xte.csv")
    data = (data-data.min())/(data.max()-data.min()) # rescale pixels

    cols, rows = 4,4
    fig = plt.figure(figsize=(8, 8))
    for i in range(cols*rows):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(data[i])
    plt.show()

def get_train_val(data_path, labels_path):
    data = load_data(data_path)
    labels = pd.read_csv(labels_path)['Prediction'].to_numpy()
    np.random.seed(0)
    val_idx = np.random.choice(np.arange(len(data)), len(data)//5, replace=False)
    mask = np.ones(len(data), dtype=bool)
    mask[val_idx] = 0

    val_data = data[val_idx]
    val_labels = labels[val_idx]

    train_data = data[mask]
    train_labels = labels[mask]

    return train_data, train_labels, val_data, val_labels