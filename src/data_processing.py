import numpy as np
import os
import matplotlib.pyplot as plt

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