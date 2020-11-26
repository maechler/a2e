from typing import List, Callable
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from tensorflow.python.distribute.multi_process_lib import multiprocessing
from a2e.automl import KerasPredictScorer, ScoreInfo
from a2e.utility import synchronized


class EstimatorSearch:
    def __init__(self, estimator: any, parameter_grid: dict, scoring: any, optimizer: str = 'bayes', cv=None, n_jobs: int = None, n_iterations: int = 50, scoring_callbacks: List[Callable] = [], **kwargs):
        self.optimizer = None
        self.estimator = estimator
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.scoring_callbacks = scoring_callbacks
        self.state = {
            'model_id': 0,
            'history': []
        }

        if isinstance(scoring, KerasPredictScorer):
            scoring.estimator_search = self

        if optimizer == 'bayes':
            # use kwargs to set optimizer_kwargs
            self.optimizer = BayesSearchCV(
                estimator=estimator,
                search_spaces=parameter_grid,
                scoring=scoring,
                cv=[(slice(None), slice(None))] if cv is None else cv,
                n_jobs=n_jobs,
                n_iter=n_iterations,
                **kwargs,
            )
        elif optimizer == 'grid':
            self.optimizer = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring=scoring,
                cv=[(slice(None), slice(None))] if cv is None else cv,
                n_jobs=n_jobs,
                **kwargs,
            )
        elif optimizer == 'random':
            self.optimizer = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=parameter_grid,
                scoring=scoring,
                cv=[(slice(None), slice(None))] if cv is None else cv,
                n_jobs=n_jobs,
                n_iter=n_iterations,
                **kwargs,
            )
        elif optimizer == 'genetic':
            # use kwargs to set population_size, gene_mutation_prob, gene_crossover_prob, tournament_size
            self.optimizer = EvolutionaryAlgorithmSearchCV(
                estimator=estimator,
                params=parameter_grid,
                scoring=scoring,
                cv=2 if cv is None else cv,
                n_jobs=n_jobs,
                generations_number=n_iterations,
                **kwargs,
            )
        else:
            raise ValueError(f'Invalid optimizer "{optimizer}" provided.')

        if n_jobs is not None and n_jobs > 10:
            print('create manager')
            process_manager = multiprocessing.Manager()
            self._lock = process_manager.RLock()
            self.state = process_manager.dict()
            self.state['model_id'] = 0
            self.state['history'] = process_manager.list()

    def fit(self, x, y=None):
        if y is None:
            y = x

        return self.optimizer.fit(x, y)

    def scoring_history(self):
        if not isinstance(self.scoring, KerasPredictScorer):
            raise ValueError(f'Scoring history is only supported for a2e.automl.KerasPredictScorer.')

        return self.state['history']

    @synchronized
    def add_score_to_history(self, score_info: ScoreInfo, estimator_sk_params: dict = {}):
        self.state['model_id'] = self.state['model_id'] + 1

        best_model_row_score = 0
        best_model_row_id = 0

        for history in self.state['history']:
            if history['score'] > best_model_row_score:
                best_model_row_score = history['score']
                best_model_row_id = history['model_id']

        history_row = {
            'model_id': self.state['model_id'],
            'best_model_id': self.state['model_id'] if score_info.score > best_model_row_score else best_model_row_id,
            'score': score_info.score,
        }

        for key, value in score_info.score_metrics.items():
            history_row[key] = value

        for key, value in estimator_sk_params.items():
            history_row[key] = value

        self.state['history'].append(history_row)

        for scoring_callback in self.scoring_callbacks:
            scoring_callback(self, score_info)

    #def refit(self, ):
    #    best_estimator = clone(base_estimator).set_params(
    #        **best_parameters)
    #    if y is not None:
    #        best_estimator.fit(X, y, **self.fit_params)
