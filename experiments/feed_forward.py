from pandas import DataFrame
from tensorflow.keras import regularizers
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.models import create_feed_forward_autoencoder
from a2e.processing.normalization import min_max_scale
from a2e.utility import build_samples

config = {
 'input_size': 32,
 'encoding_size': 16,
 'epochs': 50,
 'shuffle': True,
 'validation_split': 0.1,
 'fit_modes': ['per_feature', 'per_sample', 'train'],
 'data_columns': ['rms', 'crest', 'temperature'],
 'data_sets': ['400rpm', '800rpm', '1200rpm', 'variable_rpm'],
}
run_configs = {
 'data_column': config['data_columns'],
 'data_set': config['data_sets'],
 'fit_mode': config['fit_modes'],
}

experiment = Experiment(auto_datetime_directory=True)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)

model = create_feed_forward_autoencoder(config['input_size'], config['encoding_size'], activity_regularizer=regularizers.l1())


def run_callable(run_config: dict):
    def pre_processing(data_frame: DataFrame):
        samples = build_samples(data_frame.to_numpy().flatten(), config['input_size'])

        if run_config['fit_mode'] == 'train':
            return train_scaler.transform(samples)
        else:
            return min_max_scale(samples, fit_mode=run_config['fit_mode'])

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=run_config['data_column'])
    train_samples = build_samples(data_frames['train'].to_numpy().flatten(), config['input_size'])
    fit_mode = 'per_feature' if run_config['fit_mode'] == 'train' else run_config['fit_mode']
    train_samples, train_scaler = min_max_scale(train_samples, fit_mode=fit_mode, return_scaler=True)

    experiment.print('Fitting model')
    history = model.fit(
        train_samples,
        train_samples,
        verbose=0,
        epochs=config['epochs'],
        callbacks=experiment.callbacks(),
        validation_split=config['validation_split'],
        shuffle=config['shuffle'],
    )

    experiment.log_history(history)
    experiment.log_model(model)
    experiment.log_predictions(
        model=model,
        data_frames=data_frames,
        pre_processing=pre_processing,
    )


experiment.grid_run(run_configs, run_callable)
