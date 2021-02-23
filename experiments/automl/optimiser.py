import traceback
from typing import cast
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.model.keras import create_deep_easing_feed_forward_autoencoder
from a2e.model import KerasModel
from a2e.optimizer import create_optimizer, OptimizationResult
from a2e.utility import load_from_module
from experiments.automl.config_space import create_config_space

config = {
    'validation_split': 0.2,
    'data_column': 'fft',
    'max_iterations': 1000,
    'min_budget': 50,
    'max_budget': 50,
    'threshold_percentiles': [
        95,
        99,
    ],
}

run_configs = {
    'optimizer': [
        'BayesianRandomForrestOptimizer',
        'RandomOptimizer',
    ],
    'data_set': [
         '400rpm_v2',
         # '800rpm_v2',
         # '1200rpm_v2',
    ],
    'evaluation_function': [
        'a2e.evaluation.reconstruction_error_cost',
    ],
}

config_space = create_config_space(
    min_dropout_rate_input=.2,
    max_dropout_rate_input=.8,
    min_dropout_rate_hidden_layers=.2,
    max_dropout_rate_hidden_layers=.8,
    min_dropout_rate_output=.2,
    max_dropout_rate_output=.8,
)


if __name__ == '__main__':
    experiment = Experiment(auto_datetime_directory=True)
    experiment.log('config/config', config)
    experiment.log('config/run_configs', run_configs)
    experiment.log('config/config_space', str(config_space))

    def run_callable(run_config: dict):
        experiment.print('Loading data')
        bearing_dataset = load_data(run_config['data_set'])
        x_train = bearing_dataset.train(column=config['data_column'], as_numpy=True)

        experiment.print('Initializing optimizer')
        experiment.print(f'max_iterations = {config["max_iterations"]}')
        optimizer = create_optimizer(
            run_config['optimizer'],
            config_space=config_space,
            model=KerasModel(
                create_model_function=create_deep_easing_feed_forward_autoencoder,
                evaluation_function=load_from_module(run_config['evaluation_function']),
            ),
            x=x_train,
            max_iterations=config['max_iterations'],
            min_budget=config['min_budget'],
            max_budget=config['max_budget'],
            run_id=experiment.run_id,
            validation_split=config['validation_split'],
        )

        try:
            experiment.print('Optimizing')
            optimization_result = optimizer.optimize()
        except:
            optimization_result = OptimizationResult(optimizer.evaluation_result_aggregator.get_evaluation_results())
            experiment.log('search/error', traceback.format_exc())

        experiment.print('Logging optimization results')
        experiment.log_optimization_result(optimization_result)

        log_model_configs = [
            {'key': 'best', 'percentile_rank': 1.0},
            {'key': 'average', 'percentile_rank': 0.5},
            {'key': 'worst_10th', 'percentile_rank': 0.1},
            {'key': 'worst', 'percentile_rank': 0.0},
        ]

        for log_model_config in log_model_configs:
            keras_model = cast(KerasModel, optimizer.refit_by_percentile_rank(log_model_config['percentile_rank']))

            experiment.log_keras_model(keras_model.model, key=log_model_config['key'])

            for threshold_percentile in config['threshold_percentiles']:
                experiment.log_keras_predictions(
                    keras_model,
                    bearing_dataset.as_dict(config['data_column']),
                    key=f'{log_model_config["key"]}_{threshold_percentile}',
                    labels=bearing_dataset.as_dict(config['data_column'], labels_only=True),
                    has_multiple_features=True,
                    threshold_percentile=threshold_percentile,
                )

    experiment.multi_run(run_configs, run_callable)
