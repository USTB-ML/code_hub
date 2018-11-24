from keras.applications import ResNet50
from keras.layers.core import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.utils import plot_model


def create_Resnet50_model(nb_classes, img_rows, img_cols, weights_path=None, include_top=False, pooling=None,
                          froze=False, optimizer=None, whether_three_channel=True, is_plot_model=False,
                          loss_function='categorical_crossentropy', check_way='accuracy', plot_model_save_path=None):

    """
    :param nb_classes:
    :param img_rows:
    :param img_cols:
    :param weights_path:
    :param include_top:
    :param pooling:
    :param froze:
    :param optimizer:
    :param whether_three_channel:
    :param is_plot_model:
    :param loss_function:
    :param check_way:
    :param plot_model_save_path:
    :return:
    """

    color = 3 if whether_three_channel else 1

    base_model = ResNet50(weights=weights_path, include_top=include_top, pooling=pooling,
                          input_shape=(img_rows, img_cols, color), classes=nb_classes)

    # 冻结
    if froze == 'ALL':
        # 冻结base_model所有层
        for layer in base_model.layers:
            layer.trainable = False
    elif not froze:
        i = 0
        for layer in base_model.layers:
            if i in froze:
                layer.trainable = False
            else:
                layer.trainable = True

            i += 1

    # 是否包含top
    if not include_top:
        x = base_model.output
        x = Flatten()(x)
        # x = GlobalAveragePooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)

        # 编译
        model = Model(inputs=base_model.input, outputs=predictions)
    else:
        model = base_model

    # 编译
    model.compile(loss=loss_function, optimizer=optimizer, metrics=[check_way])

    # 绘制模型
    if is_plot_model:
        plot_model(model, to_file=plot_model_save_path, show_shapes=True)

    return model
