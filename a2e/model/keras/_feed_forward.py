import math

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD


def create_feed_forward_autoencoder(
    input_dimension,
    encoding_dimension,
    hidden_layer_activations='relu',
    output_layer_activation='sigmoid',
    optimizer='adam',
    loss='binary_crossentropy',
    activity_regularizer=None,
) -> Model:
    """Creates a feed forward Keras autoencoder model

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

    input_layer = Input(shape=(input_dimension,), name='input')
    layer = input_layer
    encoded = Dense(encoding_dimension, activation=hidden_layer_activations, activity_regularizer=activity_regularizer, name='encoded')(layer)
    layer = encoded
    output_layer = Dense(input_dimension, activation=output_layer_activation, name='output')(layer)

    model = Model(input_layer, output_layer, name='a2e_feed_forward')

    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_deep_feed_forward_autoencoder(
    input_dimension,
    number_of_hidden_layers=1,
    compression_per_layer=0.7,
    hidden_layer_activations='relu',
    output_layer_activation='sigmoid',
    optimizer='adam',
    loss='binary_crossentropy',
    activity_regularizer=None,
    activity_regularizer_factor=0.01,
    learning_rate=None,
    sgd_momentum=None,
    use_dropout=False,
    dropout_rate_input=0.2,
    dropout_rate_encoder=0.2,
    dropout_rate_decoder=0.2,
    **kwargs,
) -> Model:
    if number_of_hidden_layers % 2 == 0:
        raise ValueError(f'Number of hidden layers must be odd, "{number_of_hidden_layers}" provided.')

    input_layer = Input(shape=(input_dimension,), name='input')
    layer = input_layer

    if use_dropout:
        layer = Dropout(dropout_rate_input)(layer)

    if activity_regularizer == 'none':
        activity_regularizer = None
    elif activity_regularizer == 'l1':
        activity_regularizer = regularizers.l1(activity_regularizer_factor)
    elif activity_regularizer == 'l2':
        activity_regularizer = regularizers.l2(activity_regularizer_factor)

    number_of_encoding_layers = int((number_of_hidden_layers - 1) / 2)
    encoding_layer_dimensions = []

    for i in range(number_of_encoding_layers):
        encoding_layer_dimensions.append(int(math.pow(compression_per_layer, i+1) * input_dimension))

    # encoding
    for i, layer_dimension in enumerate(encoding_layer_dimensions):
        layer = Dense(layer_dimension, activation=hidden_layer_activations, name=f'hidden_encoding_layer_{i}')(layer)

        if use_dropout:
            layer = Dropout(dropout_rate_encoder)(layer)

    encoding_layer_size = int(math.pow(compression_per_layer, number_of_encoding_layers+1) * input_dimension)
    encoded = Dense(encoding_layer_size, activation=hidden_layer_activations, activity_regularizer=activity_regularizer, name='encoded')(layer)
    layer = encoded

    # decoding
    for i, layer_dimension in enumerate(reversed(encoding_layer_dimensions)):
        layer = Dense(layer_dimension, activation=hidden_layer_activations, name=f'hidden_decoding_layer_{i}')(layer)

        if use_dropout:
            layer = Dropout(dropout_rate_decoder)(layer)

    output_layer = Dense(input_dimension, activation=output_layer_activation, name='output')(layer)

    model = Model(input_layer, output_layer, name='a2e_deep_feed_forward')

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=sgd_momentum)

    model.compile(optimizer=optimizer, loss=loss)

    return model


def compute_model_compression(model: Model) -> float:
    input_dimension = model.input_shape[1]
    encoding_layer_index = int((len(model.layers) - 1) / 2 + 1)
    encoding_dimension = model.layers[encoding_layer_index].input_shape[1]
    compression = 1 - (encoding_dimension / input_dimension)

    return compression
