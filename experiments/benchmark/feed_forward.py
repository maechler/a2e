from sklearn.preprocessing import MinMaxScaler
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.model.keras import create_deep_easing_feed_forward_autoencoder
from a2e.processing.normalization import Scaler

config = {
    'input_dimension': 1025,
    'latent_dimension': 512,
    'hidden_layer_activations': 'relu',
    'output_layer_activation': 'relu',
    'dropout_rate_input': .2,
    'dropout_rate_hidden_layers': .2,
    'dropout_rate_output': .2,
    'activity_regularizer': 'l2',
    'loss': 'mse',
    'epochs': 50,
    'shuffle': True,
    'validation_split': 0.2,
    'data_column': 'fft',
    'threshold_percentiles': [
        95,
        99,
    ],
}
run_configs = {
    'data_set': [
        '400rpm_v2',
        '800rpm_v2',
        '1200rpm_v2',
    ],
}

experiment = Experiment(auto_datetime_directory=True)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)

model = create_deep_easing_feed_forward_autoencoder(
    input_dimension=config['input_dimension'],
    latent_dimension=config['latent_dimension'],
    hidden_layer_activations=config['hidden_layer_activations'],
    output_layer_activation=config['output_layer_activation'],
    dropout_rate_input=config['dropout_rate_input'],
    dropout_rate_hidden_layers=config['dropout_rate_hidden_layers'],
    dropout_rate_output=config['dropout_rate_output'],
    activity_regularizer=config['activity_regularizer'],
    loss=config['loss'],
)


def run_callable(run_config: dict):
    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=config['data_column'])
    labels = bearing_dataset.as_dict(column=config['data_column'], labels_only=True)
    train_scaler = Scaler(MinMaxScaler)
    train_samples = train_scaler.fit_transform(data_frames['train'].to_numpy())

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

    for percentile in config['threshold_percentiles']:
        experiment.log_keras_predictions(
            model=best_model,
            data_frames=data_frames,
            labels=labels,
            pre_processing=lambda data_frame: train_scaler.transform(data_frame.to_numpy()),
            has_multiple_features=True,
            threshold_percentile=percentile,
            key=f'{percentile}_percentile',
        )


experiment.multi_run(run_configs, run_callable)
