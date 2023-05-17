import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from PIL import Image

def map_array(n):
    if n == -1:
        return [255, 0, 0]
    if n == 0:
        return [0, 255, 0]
    if n == 1:
        return [0, 0, 255]


if __name__ == '__main__':
    df = pd.read_csv('img_dataset.csv')

    clf = svm.SVC(C=1.0, kernel='rbf')
    clf.fit(df[['r', 'g', 'b']], df['class'])

    with Image.open("images/cow.jpg") as img:
        pixels = np.asarray(img)
        full_img = pixels.reshape(pixels.shape[0] * pixels.shape[1], pixels.shape[2])

    classified = clf.predict(full_img)
    mapped = np.transpose(np.asarray(list(map(map_array, classified))))
    r = mapped[0].reshape(pixels.shape[0:2])
    g = mapped[1].reshape(pixels.shape[0:2])
    b = mapped[2].reshape(pixels.shape[0:2])

    result = np.concatenate(r, g, b, axis=2)

    plt.imshow(result)
    plt.show()
