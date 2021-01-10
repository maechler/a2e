import numpy as np
from typing import Callable, Optional, Dict, Union
from tensorflow.python.keras.models import Model
from a2e.evaluation import EvaluationResult
from a2e.model import AbstractModel
from a2e.processing.normalization import Scaler, create_scaler


class KerasModel(AbstractModel):
    def __init__(
        self,
        create_model_function: Callable[[any], Model],
        evaluation_function: Callable[..., EvaluationResult],
        fit_kwargs: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ):
        self._create_model_function = create_model_function
        self._evaluation_function = evaluation_function
        self.model: Union[Model, None] = None
        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}
        self.loaded_config = {}
        self.scaler: Union[Scaler, None] = None

        if config is not None:
            self.load_config(config)

    def create_model(self, config: dict = None, **kwargs):
        config = config if config is not None else {}

        return self._create_model_function(**config, **kwargs)

    def load_config(self, config: dict = None, **kwargs):
        config = config if config is not None else {}

        self.loaded_config = config
        self.model = self.create_model(config, **kwargs)

        if '_scaler' in config:
            self.scaler = create_scaler(config['_scaler'])
        else:
            self.scaler = create_scaler('void')

    def evaluate(self, x_train, y_train, x_valid, y_valid, budget=None, **kwargs) -> EvaluationResult:
        history = self.fit(x_train, y_train, x_valid, y_valid, budget, **self.fit_kwargs)
        y_valid_pred = self.predict(x_valid)
        evaluation_result = self._evaluation_function(self.loaded_config, y_valid, y_valid_pred, model=self.model, history=history.history)

        evaluation_result.add_info('budget', budget)

        if 'loss' in history.history:
            evaluation_result.add_info('loss', history.history['loss'][-1])

        if 'val_loss' in history.history:
            evaluation_result.add_info('val_loss', history.history['val_loss'][-1])

        return evaluation_result

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, budget=None, **kwargs):
        budget = int(budget) if budget is not None else self.fit_kwargs['epochs']

        x_train_scaled = self.scaler.fit_transform(x_train)
        y_train_scaled = self.scaler.fit_transform(y_train)

        if x_valid is not None and y_valid is not None:
            x_valid_scaled = self.scaler.fit_transform(x_valid)
            y_valid_scaled = self.scaler.fit_transform(y_valid)
        else:
            x_valid_scaled = None
            y_valid_scaled = None
            x_valid = np.array([])

        self.scaler.fit(np.concatenate((x_train, x_valid)))
        history = self.model.fit(x_train_scaled, y_train_scaled, epochs=int(budget), verbose=0, validation_data=(x_valid_scaled, y_valid_scaled), **kwargs)

        return history

    def predict(self, x, **kwargs):
        x = self.scaler.transform(x)
        prediction = self.model.predict(x, **kwargs)
        transformed_prediction = self.scaler.inverse_transform(prediction)

        return transformed_prediction
