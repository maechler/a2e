from sklearn.preprocessing import MinMaxScaler
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.processing.normalization import Scaler
from a2e.utility import build_samples, load_from_module

config = {
    'input_size': 64,
    'epochs': 50,
    'shuffle': True,
    'validation_split': 0.1,
    'model_functions': [
        'a2e.model.keras.create_conv_max_pool_autoencoder',
        'a2e.model.keras.create_conv_dense_autoencoder',
        'a2e.model.keras.create_conv_transpose_autoencoder',
    ],
    'scalings': [
        'min_max',
    ],
    'data_set_modifier_combinations': [
        {'data_set': '400rpm', 'modifier_min': 0},
        {'data_set': '800rpm', 'modifier_min': 0},
        {'data_set': '1200rpm', 'modifier_min': 0},
        # {'data_set': 'variable_rpm', 'modifier_min': -1},
        # {'data_set': 'variable_rpm', 'modifier_min': 0},
        # {'data_set': 'variable_rpm', 'modifier_min': 400},
        # {'data_set': 'variable_rpm', 'modifier_min': 800},
    ],
    'fit_modes': [
        'per_feature',
        # 'per_sample',
    ],
    'data_columns': [
        'rms',
        'crest',
        'temperature',
    ],
}
run_configs = {
 'data_column': config['data_columns'],
 'model_function': config['model_functions'],
 'data_set_modifier_combination': config['data_set_modifier_combinations'],
 'fit_mode': config['fit_modes'],
 'scaling': config['scalings'],
}

experiment = Experiment()
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    def modifier(x):
        return x[x.rpm > run_config['data_set_modifier_combination']['modifier_min']]

    def pre_processing(data_frame):
        samples = build_samples(data_frame.to_numpy().flatten(), config['input_size'], target_dimensions=3)

        if run_config['scaling'] == 'min_max':
            samples = Scaler(MinMaxScaler, fit_mode=run_config['fit_mode']).fit_transform(data_frame.to_numpy())

        return samples

    experiment.print('Building model')
    model_function = load_from_module(run_config['model_function'])
    model = model_function(config['input_size'])
    experiment.log_keras_model(model)

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set_modifier_combination']['data_set'])
    data_frames = bearing_dataset.as_dict(column=run_config['data_column'], modifier=modifier, split_test=True)
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
        # labels=bearing_dataset.as_dict(run_config['data_column'], labels_only=True),
    )


def run_id_callable(run_config: dict):
    data_column = run_config['data_column']
    model = run_config['model_function'].replace('a2e.model.keras.create_conv_', '').replace('_autoencoder', '')
    data_set = run_config['data_set_modifier_combination']['data_set']
    modifier_min = run_config['data_set_modifier_combination']['modifier_min']

    return f'{data_column}/{model}/{data_set}/{modifier_min}'


experiment.multi_run(run_configs, run_callable, run_id_callable=run_id_callable, auto_run_id=False)
