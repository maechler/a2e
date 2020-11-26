from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


class KerasEstimator(KerasRegressor):
    def __init__(self, build_fn=None, verbose=0, **sk_params):
        super().__init__(build_fn=build_fn, verbose=verbose, **sk_params)
