from tensorflow.keras import Model, Input, regularizers, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D, Dropout, Flatten, Reshape


def create_fully_connected_autoencoder(
    input_dimension,
    encoding_dimension,
    optimizer='adam',
    loss='binary_crossentropy'
) -> Model:
    """Creates a fully connected Keras autoencoder model

    Parameters
    ----------
    input_dimension : integer value for input dimension

    encoding_dimension : integer value for encoding dimension

    optimizer : (optional) defaults to 'adam'

    loss : (optional) defaults to 'binary_crossentropy'

    Returns
    -------
    model : a compiled Keras model
    """

    #input_dimension = get_property_recursively(kwargs, 'input_dimension')
    #encoding_dimension = get_property_recursively(kwargs, 'encoding_dimension')
    #optimizer = get_property_recursively(kwargs, 'optimizer', default='adam')
    #loss = get_property_recursively(kwargs, 'loss', default='binary_crossentropy')

    input_layer = Input(shape=(input_dimension,))
    layer = input_layer
    encoded = Dense(encoding_dimension, activation='relu', activity_regularizer=regularizers.l1(), name='encoded')(layer)
    layer = encoded
    output_layer = Dense(input_dimension, activation='sigmoid')(layer)

    model = Model(input_layer, output_layer)

    model.compile(optimizer=optimizer, loss=loss)

    return model


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
    activation='relu',
    padding='same',
) -> Model:
    number_of_features = 1
    input_layer = Input(shape=(input_dimension, number_of_features))

    layer = Conv1D(16, kernel_size, activation=activation, padding=padding)(input_layer)
    #layer = MaxPooling1D(2)(layer)
    layer = Dropout(rate=0.2)(layer)
    layer = Conv1D(8, kernel_size, activation=activation, padding=padding)(layer)

    layer = Flatten()(layer)
    layer = Dense(256)(layer)
    layer = Dense(512)(layer)

    #encoder = Model(input_layer, layer)

    layer = Reshape((input_dimension, 8))(layer)

    layer = Conv1D(8, kernel_size, activation=activation, padding=padding)(layer)
    layer = Dropout(rate=0.2)(layer)
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
) -> Model:
    number_of_features = 1

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
