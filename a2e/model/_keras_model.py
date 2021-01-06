from typing import Callable, Optional, Dict, Union
from tensorflow.python.keras.models import Model
from a2e.evaluation import EvaluationResult
from a2e.model import AbstractModel


class KerasModel(AbstractModel):
    def __init__(
        self,
        create_model_function: Callable[[any], Model],
        evaluation_function: Callable[..., EvaluationResult],
        fit_kwargs: Optional[Dict] = None,
    ):
        self._create_model_function = create_model_function
        self._evaluation_function = evaluation_function
        self.model: Union[Model, None] = None
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    def create_model(self, configuration: dict = None, **kwargs):
        configuration = configuration if configuration is not None else {}

        return self._create_model_function(**configuration, **kwargs)

    def load_config(self, config: dict = None, **kwargs):
        config = config if config is not None else {}

        self.model = self.create_model(config, **kwargs)

    def evaluate(self, config, x_train, y_train, x_valid, y_valid, budget=None, **kwargs) -> EvaluationResult:
        history = self.fit(config, x_train, y_train, x_valid, y_valid, budget, **self.fit_kwargs)
        y_valid_pred = self.predict(x_valid)
        evaluation_result = self._evaluation_function(config, y_valid, y_valid_pred, model=self.model, history=history.history)

        evaluation_result.add_info('budget', budget)

        if 'loss' in history.history:
            evaluation_result.add_info('loss', history.history['loss'][-1])

        if 'val_loss' in history.history:
            evaluation_result.add_info('val_loss', history.history['val_loss'][-1])

        return evaluation_result

    def fit(self, config, x_train, y_train, x_valid=None, y_valid=None, budget=None, **kwargs):
        x_train, x_valid = self.pre_process(config, x_train, x_valid)
        budget = int(budget) if budget is not None else self.fit_kwargs['epochs']

        self.load_config(config)

        return self.model.fit(x_train, y_train, epochs=int(budget), verbose=0, validation_data=(x_valid, y_valid), **kwargs)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
