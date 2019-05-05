#!/usr/bin/python
#  -*-coding:utf-8-*-

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def load_data(img):
    data = []
    # img = Image.open(file_path)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            x, y, z = img.getpixel((i, j))
            data.append([x / 256.0, y / 256.0, z / 256.0])
    return np.mat(data), m, n


def handle_pic(pic, num):
    img_data, row, col = load_data(pic)
    label = KMeans(n_clusters=num).fit_predict(img_data)
    label = label.reshape([row, col])

    variables = {}
    for i in range(num):
        variables[str(i)] = Image.new("L", (row, col))

    for i in range(row):
        for j in range(col):
            for _key, _value in variables.items():
                if str(label[i][j]) == _key:
                    _value.putpixel((i, j), int(0))
                else:
                    _value.putpixel((i, j), 255)
    return variables.values()

if __name__ == "__main__":
    pp = Image.open(r"C:\Users\LENOVO\Desktop\pandas\capt\ShangHai_1\1111.jpg")
    _list = handle_pic(pp,3)
    for i in _list:
        i.show()
