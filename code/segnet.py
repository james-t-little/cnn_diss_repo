def convolutional_layer_block(previous_layer, filter_size, kernel):
        x = Conv2D(filter_size, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(previous_layer)
        x = Dropout(0.2)(x)
        x = Conv2D(filter_size, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        return x


def segnet(img_shape, n_classes):
    inputs = Input(shape=img_shape)
    previous_layer = inputs

    concatenate_link = []
    for filter_size in [32, 64, 128, 256]:
        x = convolutional_layer_block(previous_layer, filter_size, (3,3))
        concatenate_link.append(x)
        x = BatchNormalisation()(x)
        x = MaxPooling2D((2, 2))(x)
        previous_layer = x

    concatenate_link = list(reversed(concatenate_link))
    x = convolutional_layer_block(previous_layer, 512, (3,3))
    previous_layer = x

    for count,filter_size in enumerate([256, 128, 64, 32]):
        x = Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(previous_layer)
        x = concatenate([x, concatenate_link[count]]])
        convolutional_layer_block(x, filter_size, (3,3))
        x = BatchNormalisation()(x)
        previous_layer = x

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, kernel_size=(1, 1), activation=activation)(previous_layer)

    return Model(inputs=inputs, outputs=outputs)


model = u_net((IMG_SIZE, IMG_SIZE, 3), n_classes = 3)