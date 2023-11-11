import json
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
# import tikzplotlib
from numpy import ndarray
from tqdm import tqdm


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def load_json(path: str, shape: Tuple[int, ...], num_batches: int) -> ndarray:
    """
    Load all json files containing data for one training each.
    :param path: path to folder
    :param shape: worker range x prefetch range
    :param num_batches: total number of batches that have been processed during training
    """
    file_names = [f for f in os.listdir(path) if f.endswith(".json")]
    data = np.zeros(shape)
    for file in tqdm(file_names, desc="loading data", unit="files"):
        with open(f"{path}{file}", "r", encoding="utf-8") as file:
            worker, prefetch, duration = json.load(file)
            data[worker - 1][prefetch - 1] = num_batches / duration
    return data


def plot_3d(data: ndarray) -> None:
    """
    Plot 2d data into 3d plot
    :param data: 2d ndarray with dimensions workers, prefetch factor
    """
    xdata = np.arange(data.shape[1])
    ydata = np.arange(data.shape[0])

    xmesh, ymesh = np.meshgrid(xdata, ydata)
    xmesh = xmesh.ravel()
    ymesh = ymesh.ravel()
    zmesh = np.zeros_like(xmesh)

    width = depth = 0.3

    fig = plt.figure()

    axplt = fig.add_subplot(projection='3d')
    # axplt.xlabel("Prefetch Faktor")
    # axplt.ylabel("Anzahl Worker")
    # axplt.zlabel("Batches pro Sekunde")
    axplt.bar3d(xmesh, ymesh, zmesh, width, depth, data.ravel(), shade=True, alpha=0.8)

    axplt.set_title("Batch Verarbeitung")
    axplt.set_xlabel('Prefetch Faktor')
    axplt.set_ylabel('Anzahl Worker')
    axplt.set_zlabel('Batches pro Sekunde')
    axplt.view_init(elev=20., azim=135)

    # fig = plt.gcf()
    # fig = tikzplotlib_fix_ncols(fig)
    # tikzplotlib.save('test.tex')

    plt.savefig('worker-3d.pdf')

    plt.show()

def plot_2d(data: ndarray, stds: ndarray, name: str, xlabel: str) -> None:
    """
    Plot 2d data, which has been summarized from 3d Data along one axis.
    :param data: ndarray 2d data
    :param stds: standart deviation ndarray fro each data point
    :param name: name for saved file
    :param xlabel: name for x-axis label
    """
    xdata = np.arange(data.shape[0])
    fig, axplt = plt.subplots()
    axplt.set_xlabel(xlabel)
    axplt.set_ylabel('Batches pro Sekunde')
    axplt.bar(xdata, data, align="center", yerr=stds)

    plt.savefig(name)
    plt.show()


data = load_json("logs/worker-data-11-11/", (40, 10), 79 * 5)
print(np.argmax(data))
plot_3d(data)
data_2d = np.mean(data, axis=0)
stds = np.std(data, axis=0)
plot_2d(data_2d, stds, "prefetch-2d.pdf", "Prefetch Faktor")

data_2d = np.mean(data, axis=1)
stds = np.std(data, axis=1)
plot_2d(data_2d, stds, "worker-2d.pdf", "Anzahl Worker")

