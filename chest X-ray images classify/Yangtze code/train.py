import keras
from read_tfrecord import imgs_input_fn
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import plot_model

batch_size = 32
epoch = 1
NB_CLASS = 20
IM_WIDTH = 224
IM_HEIGHT = 224


def ResNet50_model(nb_classes=15, img_rows=224, img_cols=224, RGB=False, is_plot_model=False):
    color = 3 if RGB else 1
    base_model = ResNet50(weights='.//resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                          include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                          classes=nb_classes)

    # 冻结base_model所有层
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x)

    # 训练模型
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])

    # 绘制模型
    if is_plot_model:
        plot_model(model, to_file='resnet50_model.png', show_shapes=True)

    return model


test_path = 'data_set' + str(11) + '.tfrecords'
x_test, y_test = imgs_input_fn(test_path, Run=False)
print(x_test.shape[0], 'test samples')

model = ResNet50_model()
# Let's train the model using RMSprop

model.summary()
best_acc = []

for i in range(epoch):

    for y in range(11):

        train_path = 'data_set' + str(y) + '.tfrecords'

        # 训练：
        x_train, y_train = imgs_input_fn(train_path, Run=False)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  # steps_per_epoch=10000//batch_size,
                  epochs=1,
                  # validation_data=(x_test, y_test),
                  verbose=1)

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
