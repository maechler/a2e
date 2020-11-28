from a2e.automl import EstimatorSearch, KerasEstimator, KerasPredictScorer
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.models import create_deep_feed_forward_autoencoder
from a2e.utility import load_from_module

config = {
    'input_size': 1025,
    'encoding_size': 600,
    'epochs': 15,
    'shuffle': True,
    'validation_split': 0.1,
    'data_column': 'fft',
    'data_sets': [
         '400rpm',
         '800rpm',
         #'1200rpm',
         #'variable_rpm'
    ],
    'score_functions': [
        ('a2e.automl.health_score', True),
        ('a2e.automl.min_health_score', True),
        ('a2e.automl.val_loss_score', False),
        ('a2e.automl.loss_score', False),
        ('a2e.automl.reconstruction_error_score', False),
        ('a2e.automl.f1_loss_compression_score', True),
    ]
}
run_configs = {
    'data_set': config['data_sets'],
    'score_function': config['score_functions'],
}
param_grid = {
    'epochs': [15],
    'batch_size': [50, 100, 200, 300],
    'input_dimension': [config['input_size']],
    'number_of_hidden_layers': list(range(1, 10, 2)),
    'compression_per_layer': list(map(lambda x: x/100.0, range(30, 100, 5))),
    'hidden_layer_activations': ['relu', 'linear', 'sigmoid', 'tanh'],
    'output_layer_activation': ['relu', 'linear', 'sigmoid', 'tanh'],
    #'loss': ['mse', 'binary_crossentropy'],
    # 'optimizer': ['adam'],
}

experiment = Experiment(auto_datetime_directory=True)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)
experiment.log('config/param_grid', param_grid)


def run_callable(run_config: dict):
    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    x_train = bearing_dataset.train(column=config['data_column'], as_numpy=True)

    estimator = KerasEstimator(build_fn=create_deep_feed_forward_autoencoder, validation_split=0.1)
    scorer = KerasPredictScorer(load_from_module(run_config['score_function'][0]), greater_is_better=run_config['score_function'][1])
    estimator_search = EstimatorSearch(
        estimator=estimator,
        parameter_grid=param_grid,
        scoring=scorer,
        n_jobs=1,
        n_iterations=100,
        optimizer='bayes',
        scoring_callbacks=experiment.scoring_callbacks(),
    )

    search_space_size = estimator_search.search_space_size()
    experiment.print(f'Search space size={search_space_size}')
    experiment.log('search/search_space_size', search_space_size)

    estimator_search.fit(x_train)

    history_sorted = estimator_search.search_history(sort_by_score=True)
    # Log best estimator
    best_estimator = estimator_search.fit_by_model_id(history_sorted.iloc[0]['model_id'], x_train, x_train)
    experiment.log_model(best_estimator.model, key='best_model')
    experiment.log_predictions(
        best_estimator.model,
        {
            'train': bearing_dataset.train(column=config['data_column']),
            'best_train': bearing_dataset.train(column=config['data_column']),
            'best_test': bearing_dataset.test(column=config['data_column']),
        },
        has_multiple_features=True,
    )

    # Log worst estimator
    worst_estimator = estimator_search.fit_by_model_id(history_sorted.iloc[-1]['model_id'], x_train, x_train)
    experiment.log_model(worst_estimator.model, key='worst_model')
    experiment.log_predictions(
        worst_estimator.model,
        {
            'train': bearing_dataset.train(column=config['data_column']),
            'worst_train': bearing_dataset.train(column=config['data_column']),
            'worst_test': bearing_dataset.test(column=config['data_column']),
        },
        has_multiple_features=True,
   )


experiment.multi_run(run_configs, run_callable)
