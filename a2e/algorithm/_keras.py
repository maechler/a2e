import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.engine.saving import load_model
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from a2e.algorithm import AbstractAlgorithm
from a2e.pipeline import PipelineData
from a2e.plotter import plot
from a2e.processing import health_score_sigmoid
from a2e.utility import function_from_string, to_absolute_path, out_path


class KerasFitAlgorithm(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.create_model_function = function_from_string(self.get_config('create_model_function'))
        self.create_model_function_parameters = self.get_config('create_model_function_parameters')
        self.model = None

    def init(self) -> None:
        self.model = self.create_model_function(**self.create_model_function_parameters)

    def process(self, pipeline_data: PipelineData):
        my_callbacks = [
            ModelCheckpoint(out_path('keras_model.best.hdf5'), save_best_only=True, monitor='val_loss', mode='min'),
            CSVLogger(out_path('keras_loss_log.csv'), separator=',', append=False),
        ]

        x_train, x_valid, y_train, y_valid = train_test_split(pipeline_data.algorithm_input, pipeline_data.algorithm_input, test_size=0.1, shuffle=True)

        pipeline_data.history = self.model.fit(
            x_train,
            x_train,
            validation_data=(x_valid, x_valid),
            epochs=self.get_config('epochs', default=50),
            batch_size=self.get_config('batch_size', default=500),
            shuffle=self.get_config('shuffle', default=True),
            callbacks=my_callbacks,
            verbose=self.get_config('verbose', default=True),
            validation_split=self.get_config('validation_split', default=0.2)

        )

        plot(x=range(0, len(pipeline_data.history.history['loss'])), y=pipeline_data.history.history['loss'], ylabel='Training Loss', out_name='loss', show=True, time_format=False)
        plot(x=range(0, len(pipeline_data.history.history['loss'])), y=pipeline_data.history.history['val_loss'], ylabel='Validation Loss', out_name='val_loss', show=True, time_format=False)


class KerasPredictionAlgorithm(AbstractAlgorithm):

    def __init__(self, config):
        super().__init__(config)

        self.model = None

    def init(self) -> None:
        load_model_path = to_absolute_path(self.get_config('load_model_path'))
        self.model = load_model(load_model_path)

    def process(self, pipeline_data: PipelineData):
        x_test = pipeline_data.algorithm_input
        data_frame = pipeline_data.data_set.all_data

        reconstruction = self.model.predict(x_test)
        reconstruction_difference = reconstruction - x_test
        reconstruction_error = np.sum(np.abs(reconstruction_difference), axis=1)

        data_set = DataFrame()
        data_set['reconstruction_error'] = reconstruction_error.flatten()
        data_set['reconstruction_error_rolling'] = data_set['reconstruction_error'].rolling(window=600).median()
        #data_set['reconstruction_error_rolling'] = data_set['reconstruction_error'].rolling(window=600).median()

        #rolling_error_std = data_set['reconstruction_error_rolling'].loc[pipeline_data.data_set.mask('train')].std()
        #rolling_error_median = data_set['reconstruction_error_rolling'].loc[pipeline_data.data_set.mask('train')].median()

        rolling_error_std = data_set['reconstruction_error_rolling'].std()
        rolling_error_median = data_set['reconstruction_error_rolling'].median()

        health_score = []

        for index, row in data_set.iterrows():
            current_error = row['reconstruction_error_rolling']
            magnitude_of_std = abs(current_error - rolling_error_median) / rolling_error_std
            health = health_score_sigmoid(magnitude_of_std)
            health_score.append(health)

        data_set['health_score'] = health_score

        plot(x=data_set.index, y=health_score, ylabel='Health Score', out_name='health-score', show=True, ylim=[0,1], show_screw_tightened=False, anomalous_start=pipeline_data.data_set.masks['test_anomalous']['start'], time_format=False)
        plot(x=data_set.index, y=data_set['reconstruction_error_rolling'], ylabel='Reconstruction error rolling ', out_name='reconstruction-error-rolling', show=True, show_screw_tightened=False, anomalous_start=pipeline_data.data_set.masks['test_anomalous']['start'], time_format=False)
