from pprint import pprint
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from a2e.algorithm import AbstractAlgorithm
from a2e.automl import my_make_scorer
from a2e.pipeline import PipelineData
from a2e.utility import function_from_string, out_path


class ScikitSearchAlgorithm(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.create_model_function = function_from_string(self.get_config('create_model_function'))
        self.create_model_function_parameters = self.get_config('create_model_function_parameters')
        self.model = None

    def init(self) -> None:
        if self.get_config('search_strategy', default='grid') == 'grid':
            param_grid = self.get_config('param_grid', default={})
            scoring_function = function_from_string(self.get_config('scoring_function'))
            scoring = my_make_scorer(scoring_function, greater_is_better=True)
            estimator = KerasRegressor(build_fn=self.create_model_function, verbose=2)

            self.model = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring)

    def process(self, pipeline_data: PipelineData):
        x_train = pipeline_data.algorithm_input

        my_callbacks = [
            ModelCheckpoint(out_path('scikit_model.best.hdf5'), save_best_only=True, monitor='val_loss', mode='min'),
            CSVLogger(out_path('scikit_loss_log.csv')),
        ]

        grid_result = self.model.fit(x_train, x_train, callbacks=my_callbacks, validation_split=0.2)

        pprint(grid_result.cv_results_)
        print('')
        print('')
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
