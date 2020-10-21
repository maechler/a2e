from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense


def create_feed_forward_autoencoder(
    input_dimension,
    encoding_dimension,
    optimizer='adam',
    loss='binary_crossentropy',
    activity_regularizer=None,
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

    input_layer = Input(shape=(input_dimension,), name='input')
    layer = input_layer
    encoded = Dense(encoding_dimension, activation='relu', activity_regularizer=activity_regularizer, name='encoded')(layer)
    layer = encoded
    output_layer = Dense(input_dimension, activation='sigmoid', name='output')(layer)

    model = Model(input_layer, output_layer, name='a2e_feed_forward')

    model.compile(optimizer=optimizer, loss=loss)

    return model
