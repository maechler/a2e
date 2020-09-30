from keras import Model, Input, regularizers
from keras.layers import Dense, Conv1D, MaxPooling1D, UpSampling1D
from a2e.utility import get_property_recursively


def create_fully_connected_autoencoder(input_dimension, encoding_dimension, optimizer='adam', loss='binary_crossentropy'):
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


def create_cnn_autoencoder(input_dimension, optimizer='adam', loss='binary_crossentropy'):
    input_layer = Input(shape=(input_dimension, 1))

    layer = Conv1D(16, 4, activation='relu', padding='same')(input_layer)
    layer = MaxPooling1D(4, padding='same')(layer)
    layer = Conv1D(16, 4, activation='relu', padding='same')(layer)

    encoder = Model(input_layer, layer)

    layer = Conv1D(16, 4, activation='relu', padding='same')(layer)
    layer = UpSampling1D(4)(layer)
    layer = Conv1D(16, 4, activation='relu', padding='same')(layer)

    decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(layer)

    model = Model(input_layer, decoded)

    model.compile(optimizer=optimizer, loss=loss)

    return model