from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM, RepeatVector, TimeDistributed


def create_lstm_autoencoder(
    input_dimension,
    output_dimension=None,
    units=100,
    dropout_rate=0.2,
    activation='relu',
    optimizer='adam',
    loss='mse',
    stateful=False,
    number_of_features=1,
) -> Model:
    if output_dimension is None:
        output_dimension = input_dimension

    input_layer = Input(shape=(input_dimension, number_of_features))
    layer = input_layer
    layer = LSTM(units, activation=activation, stateful=stateful, return_sequences=False)(layer)
    layer = Dropout(rate=dropout_rate)(layer)
    encoder = layer

    layer = RepeatVector(output_dimension)(layer)
    layer = Dropout(rate=dropout_rate)(layer)
    layer = LSTM(units, activation=activation, stateful=stateful, return_sequences=True)(layer)
    layer = TimeDistributed(Dense(number_of_features))(layer)
    decoder = layer

    model = Model(input_layer, decoder)

    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_lstm_to_dense_autoencoder(
    input_dimension,
    output_dimension=None,
    units=100,
    dropout_rate=0.2,
    activation='relu',
    optimizer='adam',
    loss='mse',
    stateful=False,
    number_of_features=1,
) -> Model:
    if output_dimension is None:
        output_dimension = input_dimension

    input_layer = Input(shape=(input_dimension, number_of_features))
    layer = input_layer
    layer = Dropout(rate=dropout_rate)(layer)
    layer = LSTM(units, activation=activation, stateful=stateful, return_sequences=True)(layer)
    layer = Dropout(rate=dropout_rate)(layer)
    layer = LSTM(units, activation=activation, stateful=stateful, return_sequences=False)(layer)
    encoder = layer

    layer = Dense(output_dimension)(layer)
    decoder = layer

    model = Model(input_layer, decoder)

    model.compile(optimizer=optimizer, loss=loss)

    return model
