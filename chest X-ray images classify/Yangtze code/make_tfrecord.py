from PIL import Image
import os
import tensorflow as tf

path_pic = 'D://大学//大学//大二上//308实验室//肺片分类//images'
path_csv = r'./data_set0.csv'
cwd = os.getcwd()
train_list = []
test_list = []
num_classes = 15
items_add = 0


def int_2_one_hot(labels):
    r = []
    for i in range(num_classes):
        if labels[i] == '1':
            r.append(1)
        else:
            r.append(0)
    return r


# make tf_record
def image_2_tfrecords(listname, tf_record_path):
    tf_write = tf.python_io.TFRecordWriter(tf_record_path)
    for i in range(len(listname)):
        item = listname[i]
        item = item.strip('\n')
        items = item.split(',')
        image_name = items[0]
        image_path = os.path.join(path_pic, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image = image.resize((224, 224))
            image = image.convert('L')
            image = image.tobytes()
            labels = int_2_one_hot(items[1])
            if i % 2000 == 0:
                print(i, image_path, labels)
            features = {}
            features['raw_image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
            tf_features = tf.train.Features(feature=features)
            example = tf.train.Example(features=tf_features)
            tf_serialized = example.SerializeToString()
            tf_write.write(tf_serialized)
        else:
            print("not:", image_path)
    tf_write.close()


for y in range(12):

    train_list = []

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    print('begin to make the ' + str(y) + ' data set tfrecord file')
    path_csv = './data_set' + str(y) + '.csv'

    with open(path_csv) as csv:
        i = 0
        alllines = csv.readlines()
        for line in alllines[1:]:
            train_list.append(line)
            i += 1

    items_add = items_add + len(train_list)

    image_2_tfrecords(train_list, './data_set' + str(y) + '.tfrecords')

    print(str(y) + ' data set tfrecord file has successfully saved')
    print('There are ' + str(items_add) + ' items have been encode in all')
