import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import svm
from sklearn.metrics import confusion_matrix

CIELO_CLASS = -1
PASTO_CLASS = 0
VACA_CLASS = 1

PATH = "out/"


def build_dataset():
    imgs = np.empty(shape=(0, 4))

    with Image.open("images/cielo.jpg") as img:
        imgs = np.append(imgs, prepare_images(
            np.asarray(img), CIELO_CLASS), axis=0)

    with Image.open("images/pasto.jpg") as img:
        imgs = np.append(imgs, prepare_images(
            np.asarray(img), PASTO_CLASS), axis=0)

    with Image.open("images/vaca.jpg") as img:
        imgs = np.append(imgs, prepare_images(
            np.asarray(img), VACA_CLASS), axis=0)

    df = pd.DataFrame(imgs, columns=['r', 'g', 'b', 'class'], dtype=int)
    df.to_csv("img_dataset.csv", encoding='utf-8', index=False, header=True)
    return df


def prepare_images(pixels, classification):
    # Flatten the RGB matrix to 3 arrays
    pixels = pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2])

    # class is the same length
    class_col = np.ones(shape=(pixels.shape[0], 1)) * classification
    pixels = np.append(pixels, class_col, axis=1)
    return pixels


def partition_dataset(df, partition_percentage):
    df = df.sample(frac=1).reset_index(drop=True)

    partition_size = int(np.floor(len(df) * partition_percentage))
    partitions = []

    bottom = 0
    up = partition_size
    while bottom < len(df):

        partitions.append(df[bottom:up].copy())
        bottom += partition_size
        up += partition_size
        if up > len(df):
            up = len(df)

    if (up - bottom) != partition_size:
        partitions[-2] = pd.concat([partitions[-2],
                                   partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions


def save_heatmap(df, kernel, c):
    plt.clf()
    cmap = sns.color_palette("light:b", as_cmap=True, n_colors=5)

    ax = sns.heatmap(df, cmap=cmap,
                     annot=True, fmt=".2%", xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])

    cbar = ax.collections[0].colorbar
    tick_labels = cbar.ax.get_yticklabels()
    tick_values = cbar.get_ticks()
    for i, tick_label in enumerate(tick_labels):
        tick_label.set_text(f"{int(tick_values[i] * 100)}%")
    cbar.ax.set_yticklabels(tick_labels)

    title = "Matriz de confusión C=" + str(c) + " Kernel=" + kernel
    ax.set_title(title, fontsize=7, pad=10)
    plt.tight_layout()

    file_name = f'svm_{kernel}_{c}'
    plt.savefig(PATH + f'{file_name}.png')


if __name__ == '__main__':
    df = build_dataset()

    partitions = partition_dataset(df, 0.2)

    test = partitions[-1]
    train = pd.concat([p for p in partitions[0:-1]])

    best = {}
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    step = 0.2
    c = 0.1

    for c in np.arange(0.1, 2.0 + step, step):
        for kernel in kernels:
            print("C:", c)
            print("Kernel:", kernel)
            clf = svm.SVC(C=c, kernel=kernel)
            clf.fit(train[['r', 'g', 'b']], train['class'])
            y_pred = clf.predict(test[['r', 'g', 'b']])

            cm = confusion_matrix(test['class'], y_pred)
            cm = (cm.T / cm.sum(axis=1)).T  # Divide rows by sum of row
            save_heatmap(cm, kernel, c)

            accuracy = (cm.diagonal().sum()) / cm.sum()
            if not best or best['accuracy'] < accuracy:
                best['accuracy'] = accuracy
                best['c'] = c
                best['kernel'] = kernel
                best['clf'] = clf

    print(best)
