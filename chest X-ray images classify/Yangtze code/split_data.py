"""
此文件用来分割数据集
可以用将一个大数据集划分为多个小数据集
"""

import pandas as pd
import numpy as np


def statistics_labels(label_list):

    label_all = 0

    label_statistics = []
    label_length = len(label_list[0])-1

    for i in range(label_length):
        label_statistics.append(0)

    for label in label_list:
        for i in range(label_length):
            if label[i] == '1':
                label_statistics[i] += 1
            else:
                label_statistics[i] += 0
        label_all += 1

    print('There are ' + str(label_all) + ' data items in all')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for label in range(label_length):
        print('the ' + str(label) + ' label has come out ' + str(label_statistics[label]) + ' times')

    return label_statistics


def split_data(nb_split_data, label_path, csv_col_value, csv_col_name):

    label_frame = pd.read_csv(label_path)
    name_list = label_frame[csv_col_name].values
    label_value = label_frame[csv_col_value].values

    label_statistics = statistics_labels(label_value)

    label_need = []

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for label in range(len(label_statistics)):
        print('The ' + str(label) + ' label will come out ' +
              str(int(label_statistics[label]/nb_split_data)) + ' times in each small data set')
        label_need.append(int(label_statistics[label]/nb_split_data))

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('begin to split data set')
    image_list_for_each_data_set = []
    statistics_for_each_date_set = []
    length_list = []

    for i in range(nb_split_data):
        image_list_for_each_data_set.append([])
        statistics_for_each_date_set.append([])
        length_list.append(0)
        for x in range(len(label_statistics)):
            statistics_for_each_date_set[i].append(0)

    for i in range(len(name_list)):
        label_one = []
        flag_ = False
        add_ok = 0

        # 获取标签为1的位置
        for x in range(len(label_value[i])):
            if label_value[i][x] == '1':
                label_one.append(x)

        # 将图片放入data set中
        for y in range(nb_split_data):

            # 判断该data set是否缺少此标签的数据
            for one in label_one:
                if statistics_for_each_date_set[y][one] < label_need[one]:
                    add_ok += 1

            # 如果是，则将该数据计入该data set
            if add_ok == len(label_one):

                # 更新数据集
                image_list_for_each_data_set[y].append([name_list[i], label_value[i]])
                length_list[y] += 1
                flag_ = True

                # 更新统计
                for z in label_one:
                    statistics_for_each_date_set[y][z] += 1

            if flag_:
                break

        if not flag_:
            min_list = min(length_list)
            for q in range(nb_split_data):
                if length_list[q] == min_list:

                    image_list_for_each_data_set[q].append([name_list[i], label_value[i]])
                    length_list[q] += 1
                    flag_ = True

                    for z in label_one:
                        statistics_for_each_date_set[q][z] += 1

                if flag_:
                    break

        if i % 5000 == 0:
            print(str(i) + ' items have been split')

    num_data_split = 0
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(nb_split_data):
        print('The ' + str(i) + ' data set has ' + str(length_list[i]) + ' items')
        print('The ' + str(i) + ' data set structure is: \n ' + str(statistics_for_each_date_set[i]))
        print('------------------------------------------------------------')
        num_data_split = num_data_split + length_list[i]

    print('There are ' + str(num_data_split) + ' items been split')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('begin to save data set')

    for i in range(nb_split_data):
        name = []
        label = []
        for data in image_list_for_each_data_set[i]:
            name.append(data[0])
            label.append(data[1])

        a = np.column_stack((name, label))
        resultLabel = pd.DataFrame(data=a, columns=['name', 'label'])
        resultLabel.to_csv('./' + "data_set" + str(i) + '.csv', index=False)
        print('save data set ' + str(i) + ' successfully')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('process over')


if __name__ == '__main__':
    split_data(12, 'Label.csv', 'label', 'name')
