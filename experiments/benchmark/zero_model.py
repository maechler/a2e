from numpy import percentile
from a2e.datasets.bearing import load_data
from a2e.experiment import Experiment
from a2e.processing.stats import mad
from a2e.utility import z_score as compute_z_score, compute_classification_metrics, compute_roc

run_configs = [
    {'data_set': '400rpm_v2'},
    {'data_set': '800rpm_v2'},
    {'data_set': '1200rpm_v2'},
]

experiment = Experiment(auto_datetime_directory=False)
experiment.log('config/run_configs', run_configs)


def run_callable(run_config: dict):
    experiment.print('Loading data')
    data_set_key = run_config['data_set']
    bearing_dataset = load_data(data_set_key)
    train = bearing_dataset.train('fft')
    test = bearing_dataset.test('fft', add_label=True)

    train_error = train.sum(axis=1)
    train_error_median = train_error.median()
    train_error_mad = mad(train_error)
    train_z_scores = list(map(lambda x: compute_z_score(x, train_error_median, train_error_mad), train_error))
    zscore_threshold = percentile(train_z_scores, 99)

    experiment.print('Predicting test dataset')
    prediction = []
    prediction_zscores = []

    for index, row in test.iterrows():
        is_anomaly = False
        z_score = compute_z_score(float(row.iloc[:-1].sum()), train_error_median, train_error_mad)
        is_anomaly = is_anomaly or (z_score > zscore_threshold)

        prediction.append(1 if is_anomaly else 0)
        prediction_zscores.append(z_score)

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
