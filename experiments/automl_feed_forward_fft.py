from typing import cast
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.model.keras import create_deep_feed_forward_autoencoder
from a2e.model import KerasModel
from a2e.optimizer import create_optimizer
from a2e.utility import load_from_module
import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


config = {
    'validation_split': 0.2,
    'data_column': 'fft',
}

run_configs = {
    'data_set': [
         #'400rpm',
         #'800rpm',
         '800rpm_gradual',
         #'1200rpm',
         #'variable_rpm'
    ],
    'optimizer': [
        # 'BayesianGaussianProcessOptimizer',
        'BayesianRandomForrestOptimizer',
        # 'BayesianHyperbandOptimizer',
        # 'HyperbandOptimizer',
        #'RandomOptimizer',
    ],
    'evaluation_function': [
        #'a2e.evaluation.reconstruction_error_cost',
        #'a2e.evaluation.health_score_cost',
        #'a2e.evaluation.min_health_score_cost',
        #'a2e.evaluation.keras.loss_cost',
        #'a2e.evaluation.keras.val_loss_cost',
        #'a2e.evaluation.keras.val_loss_vs_compression_cost',
        'a2e.evaluation.keras.reconstruction_error_vs_compression_cost',
    ],
}

configuration_space = cs.ConfigurationSpace(seed=1234)

configuration_space.add_hyperparameters([
    # csh.CategoricalHyperparameter('_pre_processing', [
    #     lambda x: x,
    #     min_max_scale,
    #     partial(min_max_scale, fit_mode='per_sample'),
    # ]),
    csh.Constant('input_dimension', value=1025),
    csh.CategoricalHyperparameter('number_of_hidden_layers', list(range(1, 10, 2))),
    csh.UniformFloatHyperparameter('compression_per_layer', lower=0.3, upper=0.9, default_value=0.7),
    csh.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1, default_value=1e-2),
    csh.CategoricalHyperparameter('hidden_layer_activations', ['relu', 'linear', 'sigmoid', 'tanh']),
    csh.CategoricalHyperparameter('output_layer_activation', ['relu', 'linear', 'sigmoid', 'tanh']),
    csh.CategoricalHyperparameter('loss', ['mse', 'binary_crossentropy']),
])

cs_sgd_momentum = csh.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9)
cs_optimizer = csh.CategoricalHyperparameter('optimizer', ['adam', 'sgd'])

configuration_space.add_hyperparameters([cs_optimizer, cs_sgd_momentum])
configuration_space.add_condition(cs.EqualsCondition(cs_sgd_momentum, cs_optimizer, 'sgd'))


if __name__ == '__main__':
    experiment = Experiment(auto_datetime_directory=True)
    experiment.log('config/config', config)
    experiment.log('config/run_configs', run_configs)
    experiment.log('config/configuration_space', str(configuration_space))

    def run_callable(run_config: dict):
        experiment.print('Loading data')
        bearing_dataset = load_data(run_config['data_set'])
        x_train = bearing_dataset.train(column=config['data_column'], as_numpy=True)

        experiment.print('Initializing optimizer')
        optimizer = create_optimizer(
            run_config['optimizer'],
            configuration_space=configuration_space,
            model=KerasModel(
                create_model_function=create_deep_feed_forward_autoencoder,
                evaluation_function=load_from_module(run_config['evaluation_function']),
            ),
            x=x_train,
            max_iterations=10,
            min_budget=5,
            max_budget=5,
            run_id=experiment.run_id,
            validation_split=config['validation_split'],
        )

        experiment.print('Optimizing')
        optimization_result = optimizer.optimize()

        experiment.print('Logging optimization results')
        experiment.log_optimization_result(optimization_result)

        log_model_configs = [
            {'key': 'best', 'percentile_rank': 1.0},
            {'key': 'average', 'percentile_rank': 0.5},
            {'key': 'worst', 'percentile_rank': 0.0},
        ]

        for log_model_config in log_model_configs:
            keras_model = cast(KerasModel, optimizer.refit_by_percentile_rank(log_model_config['percentile_rank']))

            experiment.log_keras_model(keras_model.model, key=log_model_config['key'])
            experiment.log_keras_predictions(
                keras_model.model,
                bearing_dataset.as_dict(config['data_column']),
                key=log_model_config['key'],
                labels=bearing_dataset.as_dict(config['data_column'], labels_only=True),
                has_multiple_features=True,
            )

    experiment.multi_run(run_configs, run_callable)
