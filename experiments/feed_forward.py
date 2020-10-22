import numpy as np
from tensorflow.keras import regularizers
from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.models import create_feed_forward_autoencoder
from a2e.processing.normalization import min_max_scale

config = {
 'input_size': 1025,
 'encoding_size': 600,
 'epochs': 5,
 'dataset': '800rpm',
 'data_column': 'fft',
 'fit_mode': 'per_feature',
 'shuffle': True,
 'validation_split': 0.1,
 'columns': ['fft', 'rms', 'crest'],
 'data_sets': ['400rpm', '800rpm', '1200rpm', 'variable_rpm'],
}
experiment = Experiment(auto_datetime_directory=False)
model = create_feed_forward_autoencoder(config['input_size'], config['encoding_size'], activity_regularizer=regularizers.l1())

bearing_dataset = load_data(config['dataset'])
train_samples = bearing_dataset.train(column=config['data_column'])
test_samples = bearing_dataset.test(column=config['data_column'])
all_samples = bearing_dataset.all(column=config['data_column'])

train_samples = min_max_scale(train_samples, fit_mode=config['fit_mode'])
test_samples = min_max_scale(test_samples, fit_mode=config['fit_mode'])
all_samples = min_max_scale(all_samples, fit_mode=config['fit_mode'])

history = model.fit(
 train_samples,
 train_samples,
 epochs=config['epochs'],
 callbacks=experiment.callbacks(),
 validation_split=config['validation_split'],
 shuffle=config['shuffle'],
)

reconstruction_all = model.predict(all_samples, verbose=1)
reconstruction_difference_all = reconstruction_all - all_samples
reconstruction_error_all = np.sum(np.abs(reconstruction_difference_all), axis=1)

reconstruction_test = model.predict(test_samples, verbose=1)
reconstruction_difference_test = reconstruction_test - test_samples
reconstruction_error_test = np.sum(np.abs(reconstruction_difference_test), axis=1)

experiment.log('config', config)
experiment.log_history(history)
experiment.log_model(model)
experiment.log_plot('metrics/reconstruction_error_all', x=bearing_dataset.all(column=config['data_column'], as_numpy=False).index, y=reconstruction_error_all, time_formatting=True)
experiment.log_plot('metrics/reconstruction_error_test', x=bearing_dataset.test(column=config['data_column'], as_numpy=False).index, y=reconstruction_error_test, time_formatting=True)
