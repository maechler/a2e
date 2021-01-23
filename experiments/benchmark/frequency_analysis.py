from numpy import percentile
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.processing.stats import mad
from a2e.utility import z_score as compute_z_score, compute_classification_metrics, compute_roc

config = {
    # See https://medias.schaeffler.us/en/product/rotary/rolling-and-plain-bearings/ball-bearings/deep-groove-ball-bearings/6205-c/p/351057#Product%20Information
    'defect_frequency_orders': {
        'BSFF': 2.3220,
        'BPFFO': 3.5722,
        'RPFFB': 4.6439,
        'BPFFI': 5.4278,
    },
    'motor_diameter': 9.7,
    'bearing_diameter': 8.3,
    'frequency_bandwidth': 5,
    #'zscore_threshold': 3,
    'zscore_threshold_percentile': 99,
    'colors': [
        '#D4373E',
        '#FFA039',
        '#3BCB69',
        '#7D11CD',
        '#DC72FF',
        '#3B90C3',
    ],
    'show_bandwidth': False,
}

run_configs = [
    {'data_set': '400rpm', 'rpm': 400},
    {'data_set': '400rpm_gradual', 'rpm': 400},
    {'data_set': '800rpm', 'rpm': 800},
    {'data_set': '800rpm_gradual', 'rpm': 800},
    {'data_set': '1200rpm', 'rpm': 1200},
]

experiment = Experiment(auto_datetime_directory=False)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    pulley_ratio = config['motor_diameter'] / config['bearing_diameter']
    shaft_frequency = (run_config['rpm'] / 60) * pulley_ratio
    bandwidth = config['frequency_bandwidth']
    data_set_key = run_config['data_set']
    defect_frequencies = []
    vlines = []

    experiment.print('Loading data')
    bearing_dataset = load_data(data_set_key)
    train = bearing_dataset.train('fft')
    test = bearing_dataset.test('fft', add_label=True)
    test_healthy, test_anomalous = bearing_dataset.test('fft', split='2fold')

    for i, (defect_type, defect_frequency_order) in enumerate(config['defect_frequency_orders'].items(), 0):
        defect_frequency = int(defect_frequency_order * shaft_frequency)
        train_means = train.iloc[:, defect_frequency - bandwidth:defect_frequency + bandwidth].median(axis=1)
        train_means_median = train_means.median()
        train_means_mad = mad(train_means)
        train_z_scores = list(map(lambda x: compute_z_score(x, train_means_median, train_means_mad), train_means))

        defect_frequencies.append({
            'frequency': defect_frequency,
            'train_median': train_means_median,
            'train_mad': train_means_mad,
            'train_z_scores': train_z_scores,
        })

        vlines.append({'x': defect_frequency, 'color': config['colors'][i], 'label': f'{defect_type}: {str(defect_frequency)}Hz', 'linestyle': 'dashed'})

        if config['show_bandwidth']:
            vlines.append({'x': defect_frequency + bandwidth, 'color': config['colors'][i], 'label': '', 'linestyle': 'dashed'})
            vlines.append({'x': defect_frequency - bandwidth, 'color': config['colors'][i], 'label': '', 'linestyle': 'dashed'})

    experiment.print('Plotting data')
    train_median = train.median()
    test_healthy_median = test_healthy.median()
    test_anomalous_median = test_anomalous.median()
    ylim = [0, max(max(test_healthy_median), max(test_anomalous_median), max(train_median)) + 0.1]

    experiment.plot(y=train_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (train)', key='train_' + data_set_key)
    experiment.plot(y=test_healthy_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (healthy)', key='healthy_' + data_set_key)
    experiment.plot(y=test_anomalous_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (anomalous)', key='anomalous_' + data_set_key)

    experiment.print('Predicting test dataset')
    prediction = []
    prediction_zscores = []

    for index, row in test.iterrows():
        is_anomaly = False
        largest_zscore = 0

        for defect_frequency in defect_frequencies:
            zscore_threshold = config['zscore_threshold'] if 'zscore_threshold' in config else percentile(defect_frequency['train_z_scores'], config['zscore_threshold_percentile'])
            row_mean = row.iloc[defect_frequency['frequency'] - bandwidth:defect_frequency['frequency'] + bandwidth].mean()
            z_score = compute_z_score(row_mean, defect_frequency['train_median'], defect_frequency['train_mad'])
            is_anomaly = is_anomaly or (z_score > zscore_threshold)
            largest_zscore = largest_zscore if largest_zscore > z_score else z_score

        prediction.append(1 if is_anomaly else 0)
        prediction_zscores.append(largest_zscore)

    experiment.plot(y=prediction_zscores, xlabel='time',  ylabel='z-score', key='prediction_zscores_' + data_set_key)
    experiment.plot(y=prediction, xlabel='time',  ylabel='is anomaly', label='prediction', key='prediction_' + data_set_key, close=False)
    experiment.plot(y=test['label'], label='truth', key='prediction_' + data_set_key, create_figure=False)

    roc = compute_roc(test['label'], prediction_zscores)
    metrics = compute_classification_metrics(test['label'], prediction)
    metrics['auc'] = roc['auc']

    experiment.log('roc', roc, to_pickle=True)
    experiment.log('metrics', metrics)
    experiment.plot_roc('roc', roc['fpr'], roc['tpr'])
    experiment.print(f'metrics: accuracy={metrics["accuracy"]}, precision={metrics["precision"]}, recall={metrics["recall"]}, f_score={metrics["f_score"]}, auc={roc["auc"]}')


experiment.multi_run(run_configs, run_callable)
