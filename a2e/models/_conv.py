from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dropout, Flatten, Reshape


def create_conv_max_pool_autoencoder(
    input_dimension,
    optimizer='adam',
    loss='binary_crossentropy',
) -> Model:
    input_layer = Input(shape=(input_dimension, 1))

    layer = Conv1D(32, 5, activation='relu', padding='same')(input_layer)
    layer = MaxPooling1D(2, padding='same')(layer)
    layer = Conv1D(16, 5, activation='relu', padding='same')(layer)
    layer = MaxPooling1D(2, padding='same')(layer)

    #encoder = Model(input_layer, layer)

    layer = UpSampling1D(2)(layer)
    layer = Conv1D(16, 5, activation='relu', padding='same')(layer)
    layer = UpSampling1D(2)(layer)
    layer = Conv1D(32, 5, activation='relu', padding='same')(layer)

    decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(layer)

    model = Model(input_layer, decoded)

    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_conv_dense_autoencoder(
    input_dimension,
    optimizer='adam',
    loss='binary_crossentropy',
    kernel_size=4,
    dropout_rate=0.2,
    activation='relu',
    padding='same',
    number_of_features=1,
) -> Model:
    input_layer = Input(shape=(input_dimension, number_of_features))

    layer = Conv1D(16, kernel_size, activation=activation, padding=padding)(input_layer)
    #layer = MaxPooling1D(2)(layer)
    layer = Dropout(rate=dropout_rate)(layer)
    layer = Conv1D(8, kernel_size, activation=activation, padding=padding)(layer)

    layer = Flatten()(layer)
    layer = Dense(int((input_dimension*8)/2))(layer)
    layer = Dense(input_dimension*8)(layer)

    #encoder = Model(input_layer, layer)

    layer = Reshape((input_dimension, 8))(layer)

    layer = Conv1D(8, kernel_size, activation=activation, padding=padding)(layer)
    layer = Dropout(rate=dropout_rate)(layer)
    #layer = UpSampling1D(2)(layer)
    layer = Conv1D(16, kernel_size, activation=activation, padding=padding)(layer)

    decoded = Conv1D(1, 1, activation='sigmoid', padding=padding)(layer)

    model = Model(input_layer, decoded)

    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_conv_transpose_autoencoder(
    input_dimension,
    optimizer='adam',
    loss='binary_crossentropy',
    kernel_size=7,
    dropout_rate=0.2,
    strides=2,
    activation='relu',
    padding='same',
    number_of_features=1,
) -> Model:
    model = Sequential([
        Input(shape=(input_dimension, number_of_features)),
        Conv1D(filters=32, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation),
        Dropout(rate=dropout_rate),
        Conv1D(filters=16, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation),
        Conv1DTranspose(filters=16, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation),
        Dropout(rate=dropout_rate),
        Conv1DTranspose(filters=32, kernel_size=kernel_size, padding=padding, strides=strides, activation=activation),
        Conv1DTranspose(filters=1, kernel_size=kernel_size, padding=padding),
    ])

    model.compile(optimizer=optimizer, loss=loss)

    return model
