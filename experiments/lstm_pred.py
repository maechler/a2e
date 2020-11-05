from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.models import create_lstm_autoencoder
from a2e.processing.normalization import min_max_scale
from a2e.utility import build_samples

config = {
 'input_size': 8,
 'output_size': 8,
 'prediction_shift': 8,
 'epochs': 25,
 'shuffle': True,
 'validation_split': 0.1,
 'scalings': [
     'none',
     #'min_max',
 ],
 'data_sets': [
     '400rpm',
     '800rpm',
     '1200rpm',
     'variable_rpm',
 ],
 'fit_modes': [
     'per_feature',
     #'per_sample',
 ],
 'data_columns': [
     'rms',
     #'crest',
     #'temperature',
 ],
}
run_configs = {
 'data_set': config['data_sets'],
 'data_column': config['data_columns'],
 'fit_mode': config['fit_modes'],
 'scaling': config['scalings'],
}

experiment = Experiment()
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)

output_dimension = config['input_size'] - config['prediction_shift']
model = create_lstm_autoencoder(config['input_size'], output_dimension=config['output_size'])


def run_callable(run_config: dict):
    def pre_processing_x(data_frame):
        numpy_data = data_frame.to_numpy()
        numpy_data = numpy_data[:-config['prediction_shift'], :]
        samples = build_samples(numpy_data.flatten(), config['input_size'], target_dimensions=3)

        if run_config['scaling'] is 'min_max':
            samples = min_max_scale(numpy_data, fit_mode=run_config['fit_mode'])

        return samples

    def pre_processing_y(data_frame):
        numpy_data = data_frame.to_numpy()
        numpy_data = numpy_data[config['prediction_shift']:, :]
        samples = build_samples(numpy_data.flatten(), config['output_size'], target_dimensions=3)

        if run_config['scaling'] is 'min_max':
            samples = min_max_scale(numpy_data, fit_mode=run_config['fit_mode'])

        return samples

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=run_config['data_column'], split_test=True)
    train_samples_x = pre_processing_x(data_frames['train'])
    train_samples_y = pre_processing_y(data_frames['train'])

    experiment.print('Fitting model')
    history = model.fit(
        train_samples_x,
        train_samples_y,
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
        pre_processing=pre_processing_x,
        pre_processing_y=pre_processing_y
    )


experiment.grid_run(run_configs, run_callable)
