import numpy as np
import pandas as pd
from statistics import median
from numpy import percentile
from sklearn.model_selection import train_test_split
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.model.keras import create_deep_easing_feed_forward_autoencoder
from a2e.model import KerasModel
from a2e.processing.stats import compute_reconstruction_error, mad
from a2e.utility import load_from_module, z_score, compute_classification_metrics, compute_roc
from experiments.automl.config_space import create_config_space

config = {
    'validation_split': 0.2,
    'data_column': 'fft',
    'num_evaluations': 1000,
    'budget': 50,
    'threshold_percentile': 99,
}

run_configs = {
    'data_set': [
         '400rpm_v2',
         '800rpm_v2',
         '1200rpm_v2',
    ],
    'evaluation_function': [
        'a2e.evaluation.keras.val_loss_cost',
    ],
    'config_space': [
        'all',
        'single_loss',
        'single_loss_single_scaling',
    ],
}


if __name__ == '__main__':
    experiment = Experiment(auto_datetime_directory=True)
    experiment.log('config/config', config)
    experiment.log('config/run_configs', run_configs)

    def run_callable(run_config: dict):
        if run_config['config_space'] == 'single_loss':
            config_space = create_config_space(loss=False)
        elif run_config['config_space'] == 'single_loss_single_scaling':
            config_space = create_config_space(loss=False, scaling=False)
        else:
            config_space = create_config_space()

        experiment.log('config_space', str(config_space))
        experiment.print('Loading data')
        bearing_dataset = load_data(run_config['data_set'])
        train = bearing_dataset.train(column=config['data_column'], as_numpy=True)
        test = bearing_dataset.test(column=config['data_column'], as_numpy=True)
        test_labels = bearing_dataset.test(column=config['data_column'], add_label=True)['label']
        threshold_percentile = config['threshold_percentile']
        x_train, x_valid, y_train, y_valid = train_test_split(
            train,
            train,
            test_size=config['validation_split'],
            shuffle=True,
        )
        model = KerasModel(
            create_model_function=create_deep_easing_feed_forward_autoencoder,
            evaluation_function=load_from_module(run_config['evaluation_function']),
        )
        history = pd.DataFrame(columns=['cost', 'auc', 'accuracy', 'precision', 'recall', 'f_score', 'matthews_cc'])

        for i in range(1, config['num_evaluations']):
            experiment.print(f'Evaluating configuration {i} of {config["num_evaluations"]}')
            current_config = dict(config_space.sample_configuration())

            model.load_config(current_config)
            evaluation_result = model.evaluate(x_train, y_train, x_valid, y_valid, budget=config['budget'])

            train_reconstruction_error = compute_reconstruction_error(y_train, model.predict(x_train))
            train_z_scores = z_score(train_reconstruction_error)
            anomaly_threshold = percentile(train_z_scores, threshold_percentile)

            test_prediction = model.predict(test)
            test_reconstruction_error = compute_reconstruction_error(test, test_prediction)
            test_z_scores = z_score(test_reconstruction_error, given_median=median(train_reconstruction_error), given_mad=mad(train_reconstruction_error))

            if np.isnan(np.sum(test_reconstruction_error)):
                experiment.print('Got a NaN value in test reconstruction error, skip this evaluation.')
                continue

            anomaly_prediction = (np.array(test_z_scores) > anomaly_threshold).astype(int)
            metrics = compute_classification_metrics(test_labels.values, anomaly_prediction)
            roc = compute_roc(test_labels.values, test_reconstruction_error)

            history_record = {
                'cost': evaluation_result.cost,
                'auc': roc['auc'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f_score': metrics['f_score'],
                'matthews_cc': metrics['matthews_cc'],
                **{f'info_{k}': v for k, v in evaluation_result.info.items()},
                **{f'config_{k}': v for k, v in current_config.items()}
            }

            history = history.append(history_record, ignore_index=True)

            experiment.log('history', history)


    experiment.multi_run(run_configs, run_callable)
