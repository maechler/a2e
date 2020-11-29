import pandas as pd
from itertools import product
from typing import List, Callable
from pandas import DataFrame
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from tensorflow.python.distribute.multi_process_lib import multiprocessing
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper

from a2e.automl import KerasPredictScorer, ScoreInfo
from a2e.utility import synchronized


class EstimatorSearch:
    def __init__(self, estimator: any, parameter_grid: dict, scoring: any, optimizer: str = 'bayes', cv=None, n_jobs: int = None, n_iterations: int = 50, scoring_callbacks: List[Callable] = [], **kwargs):
        self.optimizer = None
        self.estimator: BaseWrapper = estimator
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.scoring_callbacks = scoring_callbacks
        self.state = {
            'model_id': 0,
            'history': [],
            'sk_params': {},
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

        if n_jobs is not None and n_jobs > 1:
            process_manager = multiprocessing.Manager()
            self._lock = process_manager.RLock()
            self.state = process_manager.dict()
            self.state['model_id'] = 0
            self.state['history'] = process_manager.list()
            self.state['sk_params'] = process_manager.dict()

    def fit(self, x, y=None):
        if y is None:
            y = x

        return self.optimizer.fit(x, y)

    def search_history(self, sort_by_score=False) -> DataFrame:
        if not isinstance(self.scoring, KerasPredictScorer):
            raise ValueError(f'Scoring history is only supported for a2e.automl.KerasPredictScorer.')

        history_data_frame = pd.DataFrame(list(self.state['history']))

        if sort_by_score:
            ascending = self.scoring._sign < 0
            history_data_frame = history_data_frame.sort_values(by=['score'], ascending=ascending)

        return history_data_frame

    def search_space_size(self):
        return len(list(product(*self.parameter_grid)))

    def params_by_model_id(self, model_id):
        return self.state['sk_params'][model_id]

    def fit_by_model_id(self, model_id, x, y):
        sk_params = self.params_by_model_id(model_id)

        self.estimator.set_params(**sk_params)
        self.estimator.fit(x, y)

        return self.estimator

    @synchronized
    def add_score_to_history(self, score_info: ScoreInfo, estimator_sk_params: dict = {}):
        self.state['model_id'] = self.state['model_id'] + 1

        if self.scoring._sign > 0:
            best_model_row_score = 0
            best_model_row_id = 0

            for history in self.state['history']:
                if history['score'] > best_model_row_score:
                    best_model_row_score = history['score']
                    best_model_row_id = history['model_id']

            if score_info.score > best_model_row_score:
                best_model_row_id = self.state['model_id']
        else:
            best_model_row_score = float('inf')
            best_model_row_id = 0

            for history in self.state['history']:
                if history['score'] < best_model_row_score:
                    best_model_row_score = history['score']
                    best_model_row_id = history['model_id']

            if score_info.score < best_model_row_score:
                best_model_row_id = self.state['model_id']

        history_row = {
            'model_id': self.state['model_id'],
            'best_model_id': best_model_row_id,
            'score': score_info.score,
        }

        for key, value in score_info.score_metrics.items():
            history_row[key] = value

        for key, value in estimator_sk_params.items():
            history_row[key] = value

        self.state['sk_params'][self.state['model_id']] = estimator_sk_params
        self.state['history'].append(history_row)

        for scoring_callback in self.scoring_callbacks:
            scoring_callback(self, score_info)
