from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV


class EstimatorSearch:
    def __init__(self, estimator: any, parameter_grid: dict, scoring: any, optimizer: str = 'bayes', cv=None, n_jobs: int = None, n_iterations: int = 50, **kwargs):
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

    def fit(self, x, y):
        return self.optimizer.fit(x, y)

    #def refit(self, ):
    #    best_estimator = clone(base_estimator).set_params(
    #        **best_parameters)
    #    if y is not None:
    #        best_estimator.fit(X, y, **self.fit_params)
