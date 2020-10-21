from a2e.experiment import Experiment
from a2e.datasets.bearing import load_data
from a2e.model import create_feed_forward_autoencoder
import numpy as np

config = {
 'input_size': 1025,
 'encoding_size': 700,
 'epochs': 15,
 'dataset': '800rpm',
 'data_column': 'fft',
 'validation_split': 0.1,
 'columns': ['fft', 'rms', 'crest'],
 'data_sets': ['400rpm', '800rpm', '1200rpm', 'variable_rpm'],
}
experiment = Experiment(auto_datetime_directory=False)
model = create_feed_forward_autoencoder(config['input_size'], config['encoding_size'])

bearing_dataset = load_data(config['dataset'])
train_samples = bearing_dataset.train(column=config['data_column'])
test_samples = bearing_dataset.test(column=config['data_column'])
all_samples = bearing_dataset.all(column=config['data_column'])

#train_samples = min_max_scale(train_samples)
#test_samples = min_max_scale(test_samples)
#all_samples = min_max_scale(all_samples)

history = model.fit(
 train_samples,
 train_samples,
 epochs=config['epochs'],
 callbacks=experiment.callbacks(),
 validation_split=config['validation_split']
)

reconstruction = model.predict(all_samples, verbose=1)

reconstruction_difference = reconstruction - all_samples
reconstruction_error = np.sum(np.abs(reconstruction_difference), axis=1)

experiment.log('config', config)
experiment.log_history(history)
experiment.log_model(model)
experiment.log_plot('metrics/reconstruction_error', x=bearing_dataset.all(column=config['data_column'], as_numpy=False).index, y=reconstruction_error, time_formatting=True)
