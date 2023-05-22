import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from PIL import Image

PATH = "out/"


def map_array(n):
    if n == -1:
        return [0, 153, 255]
    if n == 0:
        return [0, 102, 0]
    if n == 1:
        return [102, 51, 0]


if __name__ == '__main__':
    df = pd.read_csv('img_dataset.csv')

    clf = svm.SVC(C=1.0, kernel='rbf')
    clf.fit(df[['r', 'g', 'b']], df['class'])

    file_name = "cow_f"

    with Image.open(f"images/{file_name}.jpg") as img:
        pixels = np.asarray(img)
        full_img = pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2])

    classified = clf.predict(full_img)
    mapped = np.transpose(np.asarray(list(map(map_array, classified))))
    shape = pixels.shape[0:2] + tuple([1])
    r = mapped[0].reshape(shape)
    g = mapped[1].reshape(shape)
    b = mapped[2].reshape(shape)

    result = np.concatenate((r, g, b), axis=2)

    plt.imshow(result)
    plt.savefig(PATH + f'{file_name}.png')
    plt.show()
