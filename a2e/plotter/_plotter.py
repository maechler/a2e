import os
from math import ceil
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np
import matplotlib
import matplotlib.dates as mdates
from tensorflow.keras.layers import Conv1D

style_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pretty.mplstyle')
plt.style.use(style_path)

# See https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
matplotlib.rcParams['agg.path.chunksize'] = 10000


def plot_model_layer_weights(model: Model, out_path=None, show=False):
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            if isinstance(layer, Conv1D):
                plot_conv_layer(layer, out_path=out_path, show=show)
            else:
                plot_dense_layer(layer, out_path=out_path, show=show)


def plot_dense_layer(layer, out_path=None, show=False):
    name = layer.name
    weights = layer.get_weights()[0]
    figure = plt.figure()

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.matshow(weights, fignum=figure.number)

    ax = plt.gca()
    ax.grid(False)

    figure.savefig(os.path.join(out_path, f'{name}.png'), format='png')

    if show:
        plt.show()


def plot_conv_layer(layer, out_path=None, show=False):
    name = layer.name
    weights = layer.get_weights()[0]
    figure = plt.figure()
    number_of_filters = weights.shape[2]

    for i in range(0, number_of_filters, 1):
        axes = figure.add_subplot(4, ceil(number_of_filters/4), i+1)
        filter = []

        for j in range(0, weights.shape[0], 1):
            filter.append(weights[j][0][i])

        axes.set_ylim([-0.5, 0.5])

        if i < 12:
            axes.set_xticklabels([])

        if i % 4 != 0:
            axes.set_yticklabels([])

        plt.xticks(np.arange(0, 4, 1))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.plot(filter)

    figure.savefig(os.path.join(out_path, f'{name}.png'), format='png')

    if show:
        plt.show()


def plot(
    y,
    x=None,
    color=None,
    ylabel=None,
    xlabel=None,
    label=None,
    title=None,
    ylim=None,
    xlim=None,
    vlines=[],
    time_formatting=False,
    out_path=None,
    out_formats=['png'],
    show=False,
    show_legend=True,
    create_figure=True
):
    if create_figure:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
        ax = plt.gca()

    if time_formatting:
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(30))

    if x is None:
        x = range(1, len(y) + 1)

    ax.plot(x, y, color=color, label=label)

    for vline in vlines:
        linestyle = vline['linestyle'] if 'linestyle' in vline else 'solid'

        ax.axvline(x=vline['x'], color=vline['color'], label=vline['label'], linestyle=linestyle)

    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=15)

    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=15)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_legend:
        ax.legend()

    if out_path:
        for out_format in out_formats:
            fig.savefig(out_path + '.' + out_format, format=out_format)

    if show:
        plt.show()
    else:
        plt.close()
