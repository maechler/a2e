from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment

config = {
    # See https://medias.schaeffler.us/en/product/rotary/rolling-and-plain-bearings/ball-bearings/deep-groove-ball-bearings/6205-c/p/351057#Product%20Information
    'defect_orders': {
        'BSFF': 2.3220,
        'BPFFO': 3.5722,
        'RPFFB': 4.6439,
        'BPFFI': 5.4278,
    },
    'motor_diameter': 9.7,
    'bearing_diameter': 8.3,
    'frequency_bandwidth': 5,
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
    {'data_set': '800rpm', 'rpm': 800},
    {'data_set': '1200rpm', 'rpm': 1200},
]

experiment = Experiment(auto_datetime_directory=True)
experiment.log('config/config', config)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    pully_ratio = config['motor_diameter'] / config['bearing_diameter']
    rpm = run_config['rpm']
    defect_frequencies = []
    vlines = []

    for i, (defect_type, defect_order) in enumerate(config['defect_orders'].items(), 0):
        shaft_frequency = (rpm / 60) * pully_ratio
        defect_frequency = defect_order * shaft_frequency
        defect_frequencies.append(defect_frequency)

        vlines.append({
            'x': defect_frequency,
            'color': config['colors'][i],
            'label': f'{defect_type}: {str(int(defect_frequency))}Hz',
            'linestyle': 'dashed',
        })

        if config['show_bandwidth']:
            vlines.append({
                'x': defect_frequency + config['frequency_bandwidth'],
                'color': config['colors'][i],
                'label': '',
                'linestyle': 'dashed',
            })
            vlines.append({
                'x': defect_frequency - config['frequency_bandwidth'],
                'color': config['colors'][i],
                'label': '',
                'linestyle': 'dashed',
            })

    experiment.print('Loading data')
    bearing_dataset = load_data(run_config['data_set'])
    train = bearing_dataset.train('fft')
    test_healthy, test_anomalous = bearing_dataset.test('fft', split=True)

    experiment.print('Plotting data')
    train_median = train.median()
    test_healthy_median = test_healthy.median()
    test_anomalous_median = test_anomalous.median()
    ylim = [0, max(max(test_healthy_median), max(test_anomalous_median), max(train_median)) + 0.1]

    experiment.plot(y=train_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (train)', key='train_' + run_config['data_set'])
    experiment.plot(y=test_healthy_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (healthy)', key='healthy_' + run_config['data_set'])
    experiment.plot(y=test_anomalous_median.values[:500], vlines=vlines, ylim=ylim, xlabel='frequency [Hz]',  ylabel='amplitude', title=f'{run_config["data_set"]} (anomalous)', key='anomalous_' + run_config['data_set'])


experiment.multi_run(run_configs, run_callable)
