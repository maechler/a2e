from pprint import pprint
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from a2e.automl import f1_loss_compression_score
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.models import create_feed_forward_autoencoder

config = {
 'input_size': 1025,
 'encoding_size': 600,
 'epochs': 2,
 'shuffle': True,
 'validation_split': 0.1,
 'data_column': 'fft',
 'data_sets': [
     #'400rpm',
     '800rpm',
     #'1200rpm',
     #'variable_rpm'
 ],
}
run_configs = {
 'data_set': config['data_sets'],
}
param_grid = {
    #'optimizer': ['adam'],
    'epochs': [3],
    'batch_size': [200],
    'input_dimension': [config['input_size']],
    'encoding_dimension': [1000, 800, 600, 400, 200, 50, 10],
}

experiment = Experiment(auto_datetime_directory=False)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)
experiment.log('config/param_grid', param_grid)


def run_callable(run_config: dict):
    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    x_train = bearing_dataset.train(column=config['data_column'], as_numpy=True)

    model = KerasRegressor(build_fn=create_feed_forward_autoencoder, verbose=0)
    scorer = experiment.make_scorer(f1_loss_compression_score, greater_is_better=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=[(slice(None), slice(None))], n_jobs=2)

    grid.fit(x_train, x_train, callbacks=experiment.callbacks(), validation_split=config['validation_split'])


experiment.multi_run(run_configs, run_callable)
