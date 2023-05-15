import pandas as pd
import numpy as np
from PIL import Image
from sklearn import svm

CIELO_CLASS = -1
PASTO_CLASS = 0
VACA_CLASS = 1


def build_dataset():
    imgs = np.empty(shape=(0, 4))

    with Image.open("images/cielo.jpg") as img:
        imgs = np.append(imgs, prepare_images(np.asarray(img), CIELO_CLASS), axis=0)

    with Image.open("images/pasto.jpg") as img:
        imgs = np.append(imgs, prepare_images(np.asarray(img), PASTO_CLASS), axis=0)

    with Image.open("images/vaca.jpg") as img:
        imgs = np.append(imgs, prepare_images(np.asarray(img), VACA_CLASS), axis=0)

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
        partitions[-2] = pd.concat([partitions[-2], partitions[-1]], ignore_index=True)

        partitions = partitions[:-1]

    return partitions


if __name__ == '__main__':
    df = build_dataset()

    partitions = partition_dataset(df, 0.2)

    test = partitions[-1]
    train = pd.concat([p for p in partitions[0:-1]])

    best = {}
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    step = 0.1
    c = 1.0
    # for c in np.arange(0.1, 2.0 + step, step):
    for kernel in kernels:
        print("C:", c)
        print("Kernel:", kernel)
        clf = svm.SVC(C=c, kernel=kernel)
        clf.fit(train[['r', 'g', 'b']], train['class'])
        accuracy = clf.score(test[['r', 'g', 'b']], test['class'])
        if not best or best['accuracy'] < accuracy:
            best['accuracy'] = accuracy
            best['c'] = c
            best['kernel'] = kernel
            best['clf'] = clf

    print(best)
