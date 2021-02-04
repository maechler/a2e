import traceback
from typing import cast
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.model.keras import create_deep_easing_feed_forward_autoencoder
from a2e.model import KerasModel
from a2e.optimizer import create_optimizer, OptimizationResult
from a2e.utility import load_from_module
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


config = {
    'validation_split': 0.2,
    'data_column': 'fft',
    'max_iterations': 1000,
    'min_budget': 30,
    'max_budget': 30,
}

run_configs = {
    'data_set': [
         '400rpm',
         '800rpm',
         '1200rpm',
    ],
    'optimizer': [
        # 'BayesianGaussianProcessOptimizer',
        'BayesianRandomForrestOptimizer',
        # 'BayesianHyperbandOptimizer',
        # 'HyperbandOptimizer',
        # 'RandomOptimizer',
    ],
    'evaluation_function': [
        'a2e.evaluation.keras.val_loss_cost',
        'a2e.evaluation.reconstruction_error_cost',
        'a2e.evaluation.keras.reconstruction_error_vs_compression_cost',
        'a2e.evaluation.keras.uniform_reconstruction_error_vs_compression_cost'
    ],
}

config_space = cs.ConfigurationSpace(seed=1234)

config_space.add_hyperparameters([
    csh.CategoricalHyperparameter('_scaler', [
        'none',
        'min_max',
        'std',
    ]),

    csh.UniformIntegerHyperparameter('_batch_size', lower=16, upper=512, default_value=256),
    csh.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
    csh.UniformFloatHyperparameter('learning_rate_decay', lower=1e-8, upper=1e-2, default_value=1e-4, log=True),
    csh.CategoricalHyperparameter('loss', ['mse', 'mae', 'binary_crossentropy']),

    csh.Constant('input_dimension', value=1025),
    csh.OrdinalHyperparameter('number_of_hidden_layers', list(range(1, 22, 2))),
    csh.UniformIntegerHyperparameter('latent_dimension', lower=10, upper=1024, default_value=512, log=True),
    csh.CategoricalHyperparameter('easing', ['ease_linear', 'ease_in_quad', 'ease_out_quad']),
    csh.CategoricalHyperparameter('hidden_layer_activations', ['relu', 'linear', 'sigmoid', 'tanh']),
    csh.CategoricalHyperparameter('output_layer_activation', ['relu', 'linear', 'sigmoid', 'tanh']),

    csh.UniformFloatHyperparameter('dropout_rate_input', lower=0.0, upper=0.99, default_value=0.2),
    csh.UniformFloatHyperparameter('dropout_rate_hidden_layers', lower=0.0, upper=0.99, default_value=0.2),
    csh.UniformFloatHyperparameter('dropout_rate_output', lower=0.0, upper=0.99, default_value=0.2),

    csh.CategoricalHyperparameter('activity_regularizer', ['l1', 'l2']),
    csh.UniformFloatHyperparameter('l1_activity_regularizer_factor', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
    csh.UniformFloatHyperparameter('l2_activity_regularizer_factor', lower=1e-6, upper=1e-1, default_value=1e-2, log=True),
])

# Conditions
config_space.add_condition(cs.EqualsCondition(config_space.get_hyperparameter('l1_activity_regularizer_factor'), config_space.get_hyperparameter('activity_regularizer'), 'l1'))
config_space.add_condition(cs.EqualsCondition(config_space.get_hyperparameter('l2_activity_regularizer_factor'), config_space.get_hyperparameter('activity_regularizer'), 'l2'))


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
            experiment.log_keras_predictions(
                keras_model,
                bearing_dataset.as_dict(config['data_column']),
                key=log_model_config['key'],
                labels=bearing_dataset.as_dict(config['data_column'], labels_only=True),
                has_multiple_features=True,
            )

    experiment.multi_run(run_configs, run_callable)
