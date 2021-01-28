from sklearn.preprocessing import MinMaxScaler
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.model.keras import create_feed_forward_autoencoder
from a2e.processing.normalization import Scaler

config = {
    'input_size': 1025,
    'encoding_size': 512,
    'hidden_layer_activations': 'relu',
    'output_layer_activation': 'relu',
    'loss': 'mse',
    'epochs': 30,
    'shuffle': True,
    'validation_split': 0.2,
    'data_column': 'fft',
    'threshold_percentile': 0.99,
}
run_configs = {
    'data_set': [
        '400rpm_v2',
        '800rpm_v2',
        '1200rpm_v2',
    ],
}

experiment = Experiment(auto_datetime_directory=False)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)

model = create_feed_forward_autoencoder(
    input_dimension=config['input_size'],
    encoding_dimension=config['encoding_size'],
    hidden_layer_activations=config['hidden_layer_activations'],
    output_layer_activation=config['output_layer_activation'],
    loss=config['loss'],
)


def run_callable(run_config: dict):
    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=config['data_column'])
    labels = bearing_dataset.as_dict(column=config['data_column'], labels_only=True)
    train_samples = Scaler(MinMaxScaler).fit_transform(data_frames['train'].to_numpy())

    experiment.print('Fitting model')
    history = model.fit(
        train_samples,
        train_samples,
        verbose=0,
        epochs=config['epochs'],
        callbacks=experiment.keras_callbacks(),
        validation_split=config['validation_split'],
        shuffle=config['shuffle'],
    )

    best_model = experiment.load_best_model()

    experiment.log_keras_model(best_model, key='best')
    experiment.log_history(history)
    experiment.log_keras_predictions(
        model=best_model,
        data_frames=data_frames,
        labels=labels,
        pre_processing=lambda data_frame: Scaler(MinMaxScaler).fit_transform(data_frame.to_numpy()),
        has_multiple_features=True,
        threshold_percentile=config['threshold_percentile'],
    )


experiment.multi_run(run_configs, run_callable)
