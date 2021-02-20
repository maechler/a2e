from sklearn.preprocessing import MinMaxScaler
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.processing.normalization import Scaler
from a2e.utility import build_samples, load_from_module

config = {
 'input_size': 1024,
 'epochs': 10,
 'shuffle': True,
 'validation_split': 0.1,
 'data_column': 'fft',
 'model_functions': [
     'a2e.model.keras.create_conv_max_pool_autoencoder',
     'a2e.model.keras.create_conv_dense_autoencoder',
     'a2e.model.keras.create_conv_transpose_autoencoder',
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
}
run_configs = {
 'data_set': config['data_sets'],
 'model_function': config['model_functions'],
 'scaling': config['scalings'],
 'fit_mode': config['fit_modes'],
}

experiment = Experiment()
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    def modifier(x):
        return x[x.rpm > 0]

    def pre_processing(data_frame):
        if run_config['scaling'] == 'min_max':
            samples = Scaler(MinMaxScaler, fit_mode=run_config['fit_mode']).fit_transform(data_frame.to_numpy())
        else:
            samples = data_frame.to_numpy()

        return build_samples(samples, target_sample_length=config['input_size'], target_dimensions=3)

    experiment.print('Building model')
    model_function = load_from_module(run_config['model_function'])
    model = model_function(config['input_size'])
    experiment.log_keras_model(model)

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    data_frames = bearing_dataset.as_dict(column=config['data_column'], modifier=modifier)
    train_samples = pre_processing(data_frames['train'])

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

    experiment.log_history(history)
    experiment.log_keras_model(model)
    experiment.log_keras_predictions(
        model=model,
        data_frames=data_frames,
        pre_processing=pre_processing,
        has_multiple_features=True,
    )


experiment.multi_run(run_configs, run_callable)
