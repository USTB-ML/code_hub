# TFrecord制作：
from PIL import Image
import os
import cv2
import tensorflow as tf
path_pic = r'train_save_pic'
path_csv = r'train.csv'
path_pic_test = r'test_save_pic'
path_csv_test = r'testALL.csv'
cwd = os.getcwd()
train_list = []
test_list = []
num_classes = 3

with open(path_csv) as csv:
    i = 0
    alllines = csv.readlines()
    for line in alllines[1:]:
        train_list.append(line)
        i += 1

with open(path_csv_test) as csv:
    i = 0
    alllines = csv.readlines()
    for line in alllines[1:]:
        test_list.append(line)
        i += 1


def int_2_one_hot(labels):
    r = []
    for i in range(num_classes):
        if labels[i] == '1':
            r.append(1)
        else:
            r.append(0)
    return r


def image_2_tfrecords(listname, tf_record_path, is_train=True):
    tf_write = tf.python_io.TFRecordWriter(tf_record_path)
    for i in range(len(listname)):
        item = listname[i]
        item = item.strip('\n')
        items = item.split(',')
        image_name = items[0] + ".jpg"
        if is_train:
            image_path = os.path.join(path_pic, image_name)
        else:
            image_path = os.path.join(path_pic_test, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image = image.resize((128, 128))
            image = image.tobytes()
            labels = int_2_one_hot(items[1:])
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


image_2_tfrecords(train_list, r"train.tfrecords")
image_2_tfrecords(test_list, r"valid.tfrecords", is_train=False)



------------------------------------------------------------------
# TFrecord读取：
test_path = "testTanh.tfrecords"
train_path = "trainTanh.tfrecords"

batch_size = 16
num_classes = 3
epochs = 5
num_predictions = 3
input_shape = (128, 128, 3)
train_samples = 13316
val_samples = 2000
num_div = 20


def imgs_input_fn(filenames, perform_shuffle=False, Repeats=epochs, Batchs=train_samples, Run=True):
    def _parse_function(serialized):
        features = \
            {
                'raw_image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image_shape = tf.stack([224, 224, 3])
        image_raw = parsed_example['raw_image']
        label = tf.cast(parsed_example['label'], tf.int32)
        label = tf.one_hot(label, num_classes)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32) * 1./255 - 0.5
        image = tf.reshape(image, image_shape)
        image = tf.reverse(image, axis=[2])  # 'RGB'->'BGR'
        return image, label

    # imgs, labels = _parse_function(filenames)
    # # print(imgs, labels)
    # x_batch, y_batch = K.tf.train.batch([imgs, labels], batch_size=Batchs, capacity=100)
    # print(x_batch, y_batch)
    # return x_batch, y_batch

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(Batchs)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    iterator = iterator.get_next()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        while 1:
            batch_features, batch_labels = sess.run(iterator)
            yield batch_features, batch_labels
            # if not Run:
            #     batch_features, batch_labels = iterator
            #     print(batch_features, batch_labels)
            #     return batch_features, batch_labels
            # while Run:
            #     batch_features, batch_labels = sess.run(iterator)
            #     yield {'input': batch_features}, {'output': batch_labels}



------------------------------------------------------------------
训练：
for x_, y_ in imgs_input_fn(train_path, Batchs=11316//num_div):
    x_train = x_
    y_train = y_
    break
for x_, y_ in imgs_input_fn(test_path, Batchs=500):
    x_test = x_
    y_test = y_
    break

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = keras.models.load_model('model5.h5')
# Let's train the model using RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])
model.summary()
best_acc = []
for i in range(epochs):
    turn = 0
    for x_train, y_train in imgs_input_fn(train_path, Batchs=11316//num_div):
        print('It is turn - ', turn)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  # steps_per_epoch=10000//batch_size,
                  shuffle=True,
                  epochs=1,
                  verbose=1)
        turn = turn + 1

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('epoch', i)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    # if i == 7:
    #     model.compile(loss='binary_crossentropy',
    #                   optimizer=Adam(lr=2e-5),
    #                   metrics=['accuracy'])
    if i > 0:
        best_acc.append((i, scores[1]))
        # Save the final model
        model_json = model.to_json()
        mdl_save_path = 'model'+str(i)+'.json'
        with open(mdl_save_path, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        mdl_save_path = 'model'+str(i)+'.h5'
        model.save(mdl_save_path)

print(best_acc)
