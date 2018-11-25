import tensorflow as tf

# TFrecord读取：
test_path = "testTanh.tfrecords"
train_path = "trainTanh.tfrecords"

batch_size = 32
num_classes = 15
epochs = 5
num_predictions = 3
input_shape = (224, 224, 3)
train_samples = 6000
val_samples = 6000


def imgs_input_fn(filenames, perform_shuffle=False, Repeats=epochs, Batchs=6000, Run=True):
    def _parse_function(serialized):
        features = \
            {
                'raw_image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([num_classes], tf.int64)	    # num_classes=0就空着[]
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image_shape = tf.stack([224, 224, 3])
        image_raw = parsed_example['raw_image']
        label = tf.cast(parsed_example['label'], tf.int32)
        # label = tf.one_hot(label, num_classes)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32) * 1./255 - 0.5
        image = tf.reshape(image, image_shape)
        # image = tf.reverse(image, axis=[2])  # 'RGB'->'BGR'
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
        batch_features, batch_labels = sess.run(iterator)
        return batch_features, batch_labels
        # if not Run:
        #     batch_features, batch_labels = iterator
        #     print(batch_features, batch_labels)
        #     return batch_features, batch_labels
        # while Run:
        #     batch_features, batch_labels = sess.run(iterator)
        #     yield {'input': batch_features}, {'output': batch_labels}
