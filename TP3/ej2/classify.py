import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from PIL import Image
import json
import os

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

    with open('classify_config.json') as config_file:
        config = json.load(config_file)

        kernel = config['kernel'] if 'kernel' in config else 'linear'
        c_value = config['c_value'] if 'c_value' in config else 1.6
        image_file = config['image_file'] if 'image_file' in config else 'cow_f'

        print(f"Kernel: {kernel}, C: {c_value}, Image: {image_file}")
        clf = svm.SVC(C=c_value, kernel=kernel)
        clf.fit(df[['r', 'g', 'b']], df['class'])

        with Image.open(f"images/{image_file}.jpg") as img:
            pixels = np.asarray(img)
            full_img = pixels.reshape(
                pixels.shape[0] * pixels.shape[1], pixels.shape[2])

        classified = clf.predict(full_img)
        mapped = np.transpose(np.asarray(list(map(map_array, classified))))
        shape = pixels.shape[0:2] + tuple([1])
        r = mapped[0].reshape(shape)
        g = mapped[1].reshape(shape)
        b = mapped[2].reshape(shape)

        result = np.concatenate((r, g, b), axis=2)

        plt.clf()

        plt.imshow(result)
        plt.title(f"Kernel: {kernel}, C: {c_value}")

        plt.tight_layout()

        path = PATH + "/classify"
        os.mkdir(path) if not os.path.exists(path) else None
        path += "/"+kernel
        os.mkdir(path) if not os.path.exists(path) else None

        path += f"/{str(c_value).replace('.', 'p')}_{image_file}"
        
        plt.savefig(path + '.png')
    config_file.close()
    # plt.show()
