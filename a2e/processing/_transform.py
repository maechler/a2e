import numpy as np
from sklearn.preprocessing import MinMaxScaler
from a2e.pipeline import AbstractPipelineStep, PipelineData


class FftTransformStep(AbstractPipelineStep):

    def process(self, pipeline_data: PipelineData):
        x_train = []
        data_set = pipeline_data.data_set
        data_frame = data_set.all_data if self.get_config('data_mask') == 'all' else data_set.masked_data(self.get_config('data_mask'))

        if self.get_config('filter_rpm', default=0) > 0:
            data_frame = data_frame[data_frame.rpm > self.get_config('filter_rpm')]
            pipeline_data.data_set._data_frame = data_frame  # hacky the hack

        for index, row in data_frame.iterrows():
            size = len(row['fft_magnitude'].split(','))
            fft = np.array(list(map(float, (row['fft_magnitude'].split(','))))).reshape(size, 1)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(fft)

            fft_scaled = scaler.transform(fft)
            fft_scaled_list = list(np.array(fft_scaled).reshape(1, size))[0]

            x_train.append(fft_scaled_list)

        pipeline_data.algorithm_input = np.array(x_train)


class WindowingTransformStep(AbstractPipelineStep):

    def process(self, pipeline_data: PipelineData):
        x_train = []
        sample = []
        count = 0
        data_set = pipeline_data.data_set
        data_frame = data_set.all_data

        if self.get_config('filter_rpm', default=0) > 0:
            data_frame = data_frame[data_frame.rpm > self.get_config('filter_rpm')]
            pipeline_data.data_set._data_frame = data_frame  # hacky the hack

        for value in data_frame['rms']:
            sample.append(value)
            count = count + 1

            if count % 64 == 0:
                x_train.append(sample)
                sample = []

        x_train.pop()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train)

        x_train_scaled = scaler.transform(x_train)


        pipeline_data.algorithm_input = x_train_scaled.reshape(len(x_train_scaled), 64, 1)
        #input_data.reshape((1, self.get_config('window_size'), 1))