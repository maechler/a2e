from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.processing.normalization import min_max_scale
from a2e.utility import build_samples, load_from_module

config = {
 'input_size': 128,
 'epochs': 1,
 'shuffle': True,
 'validation_split': 0.1,
 'model_functions': [
     'a2e.models.create_conv_max_pool_autoencoder',
     'a2e.models.create_conv_dense_autoencoder',
     'a2e.models.create_conv_transpose_autoencoder',
 ],
 'scalings': [
     'none',
     'min_max',
 ],
 'data_sets': [
     '400rpm',
     '800rpm',
     '1200rpm',
     'variable_rpm',
 ],
 'fit_modes': [
     'per_feature',
     'per_sample',
 ],
 'data_columns': [
     'rms',
     'crest',
     'temperature',
 ],
 'modifier_mins': [
     -1,
     0,
     400,
     800,
 ],
}
run_configs = {
 'data_set': config['data_sets'],
 'data_column': config['data_columns'],
 'fit_mode': config['fit_modes'],
 'scaling': config['scalings'],
 'model_function': config['model_functions'],
 'modifier_min': config['modifier_mins'],
}

experiment = Experiment()
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    def modifier(x):
        return x[x.rpm > run_config['modifier_min']]

    def pre_processing(data_frame):
        samples = build_samples(data_frame.to_numpy().flatten(), config['input_size'], target_dimensions=3)

        if run_config['scaling'] is 'min_max':
            samples = min_max_scale(data_frame.to_numpy(), fit_mode=run_config['fit_mode'])

        return samples

    experiment.print('Building model')
    model_function = load_from_module(run_config['model_function'])
    model = model_function(config['input_size'])
    experiment.log_model(model)

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=run_config['data_column'], modifier=modifier)
    train_samples = pre_processing(data_frames['train'])

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