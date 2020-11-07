from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.models import create_feed_forward_autoencoder
from a2e.processing.normalization import min_max_scale

config = {
 'input_size': 1025,
 'encoding_size': 600,
 'epochs': 5,
 'shuffle': True,
 'validation_split': 0.1,
 'data_column': 'fft',
 'scalings': ['none', 'min_max'],
 'data_sets': ['400rpm', '800rpm', '1200rpm', 'variable_rpm'],
 'fit_modes': ['per_feature', 'per_sample'],
}
run_configs = {
 'data_set': config['data_sets'],
 'fit_mode': config['fit_modes'],
 'scaling': config['scalings'],
}

experiment = Experiment(auto_datetime_directory=True)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)

model = create_feed_forward_autoencoder(config['input_size'], config['encoding_size'])


def run_callable(run_config: dict):
    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=config['data_column'], modifier=lambda x: x[x.rpm > 0])

    if run_config['scaling'] is 'min_max':
        train_samples = min_max_scale(data_frames['train'].to_numpy(), fit_mode=run_config['fit_mode'])
    else:
        train_samples = data_frames['train'].to_numpy()

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
        pre_processing=lambda data_frame: min_max_scale(data_frame.to_numpy(), fit_mode=run_config['fit_mode']) if run_config['scaling'] is 'min_max' else data_frame.to_numpy(),
        has_multiple_features=True,
    )


experiment.multi_run(run_configs, run_callable)
