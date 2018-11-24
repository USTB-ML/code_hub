"""
此文件用来从csv文件中读取标签并从相关图片目录中读取图片

相关函数有两个
read_image():从对应目录中按照相关要求读取图片并以np.array的形式返回图片数据
load_data_use_csv():从csv文件中读取标签、从read_image()请求图片数据并返回对应图片和标签
"""

import numpy as np
import os
import pandas as pd
import cv2


def read_image(img_name, IM_WIDTH, IM_HEIGHT, flag_colorful=True):

    """
    :param img_name: 图片名
    :param IM_WIDTH: 图片的宽度(像素)
    :param IM_HEIGHT: 图片的高度(像素)
    :param flag_colorful: 是否为3通道图片，True为是，False为单通道， 默认为True
    :return: 以np.array的形式返回图片数据
    """

    if flag_colorful:
        im = cv2.imread(img_name, 1)
    else:
        im = cv2.imread(img_name, 0)

    im = cv2.resize(im, (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_NEAREST)
    data = np.array(im)

    return data


def load_data_use_csv(image_path, label_path, csv_col_name, csv_col_value, IM_WIDTH, IM_HEIGHT, max_load=None,
                      flag_colorful=True, n_class=None):

    """
    :param image_path: 图片所在文件夹路径
    :param label_path: csv文件所在路径
    :param csv_col_name: csv文件 图片名列名
    :param csv_col_value: csv文件 标签名列名
    :param IM_WIDTH: 图片设定宽度(像素)
    :param IM_HEIGHT: 图片设定高度(像素)
    :param max_load: 最大图片加载量，默认为加载全部
    :param flag_colorful: 是否为3通道，True为是3通道，False为1通道，默认为True
    :param n_class: 总分类数，如果指定最大图片加载量，则需要传入此值，每一种类别的图片返回的数量为 int(max_load/n_class)
    :return: 以np.array的形式返回数据和标签 数据在前、标签在后
    """

    print('Begin to load data')

    print('load csv file')
    result_frame = pd.read_csv(label_path)
    result_dist = dict(zip(result_frame[csv_col_name].values, result_frame[csv_col_value].values))
    print('csv file loads successfully')

    images = []
    label = []
    image_load_every_label = []
    img_load_num = 0

    if max_load and n_class:
        for i in range(n_class):
            image_load_every_label.append(0)
        max_load_everylabel = int(max_load/n_class)

    print('Begin to load images')
    files = os.listdir(image_path)
    const_image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    for fn in files:
        if os.path.splitext(fn)[1] in const_image_format:

            if max_load and n_class:
                if image_load_every_label[int(result_dist[fn[:-4]])] <= max_load_everylabel:

                    fd = os.path.join('./train_dir', fn)
                    images.append(read_image(fd, IM_WIDTH, IM_HEIGHT, flag_colorful))
                    label.append(result_dist[fn[:-4]])
                    img_load_num += 1
                    image_load_every_label[int(result_dist[fn[:-4]])] += 1

                    if img_load_num % 1000 == 0:
                        print(img_load_num)
            else:
                fd = os.path.join('./train_dir_raw', fn)
                images.append(read_image(fd, IM_WIDTH, IM_HEIGHT, flag_colorful))
                label.append(result_dist[fn[:-4]])
                img_load_num += 1

                if img_load_num % 1000 == 0:
                    print(img_load_num)

        if max_load and n_class and img_load_num == max_load:
            break

    print('load success!')
    X = np.array(images)
    y = np.array(label)
    return X, y
